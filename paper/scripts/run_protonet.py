# reference https://github.com/oscarknagg/few-shot/blob/master/few_shot/proto.py
from ctypes import util
import sys
sys.path.append('..')
from typing import List, Tuple
import os
import time
import pickle

import torch # type: ignore
import numpy as np
import absl.flags
import absl.app
import yaml

import utils.utils as utils

# user flags
absl.flags.DEFINE_bool("continue_train", False, "std, memory or mlp")
absl.flags.DEFINE_enum("optimizer", None, ['sgd','adam'], "Optimizer to use, sgd or adam")
FLAGS = absl.flags.FLAGS


def compute_prototypes(support: torch.Tensor, targets: int, classes:int=10) -> torch.Tensor:
    """Compute class prototypes from support samples.

    Arguments:
        support: torch.Tensor. Support samples.
        targets: int. target class for each support sample.
        classes: int. number of classes in the classification task

    Returns:
        class_prototypes: Prototypes aka mean embeddings for each class
    """
    # Reshape so the first dimension indexes by class then take the mean
    # along that dimension to generate the "prototypes" for each class
    class_prototypes = []
    for label in range(classes):
        class_prototypes.append(support[targets==label].mean(dim=0))
    return torch.stack(class_prototypes)


def train_model(model:torch.nn.Module, loaders:List[torch.utils.data.DataLoader],
                optimizer:torch.optim.Optimizer, scheduler:torch.optim.lr_scheduler._LRScheduler,
                loss_criterion:torch.nn.modules.loss, num_epochs:int,
                device:torch.device) -> torch.nn.Module:
    """ Function to train a model with a Memory Wrap layer (in the paper both
    the baseline variant and Memory Wrap)

    Args:
        model (torch.nn.Module): Model with a Memory Wrap layer to be trained
        loaders (List[torch.utils.data.DataLoader]): Loaders containing
            dataset subsets to be used to train the model. The loaders[0]
            element is the training dataset, while loaders[1] contain the
            dataset used to sample memory sets
        optimizer (torch.optim.Optimizer): PyTorch optimizer to use to perform
            training step
        scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate
        scheduler to adaptive adjusting the learning rate during training
        loss_criterion (torch.nn.modules.loss): criterion to use to compute
            the loss
        num_epochs (int): number of epoch to train the model
        device (torch.device): device where the model is stored

    Returns:
        torch.nn.Module: the trained model
    """
    train_loader, mem_loader = loaders
    # training process
    model.train()

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(1, num_epochs + 1):
        for batch_idx, (data, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # input
            data = data.to(device)
            targets = targets.to(device)
            support_set, targets_memory = next(iter(mem_loader))
            support_set = support_set.to(device)
           
            with torch.cuda.amp.autocast():
                #get embedding x
                embedding_input = model(data)
                embedding_support = model(support_set)

                # compute prototypes
                sorted_targets_memory, sorted_indices = torch.sort(targets_memory)
                sorted_support = embedding_support[sorted_indices]
                prototypes = compute_prototypes(sorted_support, sorted_targets_memory)
                prototypes = prototypes.to(device)

                # get output
                distances = utils.vector_distance(embedding_input, prototypes, 'cosine')
                log_p_y = (-distances).log_softmax(dim=1)
                
                loss = loss_criterion(log_p_y, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            #log stuff:
            print(f'Train Epoch: {epoch} [{batch_idx} ' \
                    f'/{len(train_loader)}]\t',end='\r')

        scheduler.step()# increase scheduler step for each epoch

    return model


def eval_model(model:torch.nn.Module, loader:torch.utils.data.DataLoader,
               mem_loader:torch.utils.data.DataLoader, loss_criterion:torch.nn.modules.loss,
               device:torch.device)-> Tuple[float, float]:
    """ Method to evaluate a pytorch model that has a Memory Wrap variant
        as last layer. The model takes an image and a memory samples as input and return the logits.

    Args:
        model (torch.nn.Module): Trained model to evaluate
        loader (torch.utils.data.DataLoader): loader containing testing
            samples where evaluate the model
        mem_loader (torch.utils.data.DataLoader): loader containing training
            samples to be used as memory set
        loss_criterion (torch.nn.modules.loss): loss to use to evaluate the
            error
        device (torch.device): device where the model is stored

    Returns:
        Tuple[float, float]: Accuracy and loss in the given test dataset
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for _, (data, target) in enumerate(loader):
            data = data.to(device)
            target = target.to(device)
            support_set, targets_memory = next(iter(mem_loader))
            support_set = support_set.to(device)
            with torch.cuda.amp.autocast():
                # get embedding x
                embedding_input = model(data)
                embedding_support = model(support_set)
                
                # compute prototypes
                sorted_targets_memory, sorted_indices = torch.sort(targets_memory)
                sorted_support = embedding_support[sorted_indices]
                prototypes = compute_prototypes(sorted_support, sorted_targets_memory)
                prototypes = prototypes.to(device)

                # get output
                distances = utils.vector_distance(embedding_input, prototypes, 'cosine')
                log_p_y = (-distances).log_softmax(dim=1)
                pred = (-distances).softmax(dim=1)
                pred = pred.data.max(1, keepdim=True)[1]
                
                loss = loss_criterion(log_p_y, target)
            
            # log stuff
            correct += pred.eq(target.data.view_as(pred)).sum().item()
            test_loss += loss.item()

        test_accuracy = 100.*(torch.true_divide(correct,len(loader.dataset)))
    return test_accuracy,  test_loss


def run_experiment(config:dict):
    """Method to run an experiment. Each experiment is composed by n
    runs, defined in the config dictionary, where in each of them a new
    model is trained.

    Args:
        config (dict): Dictionary containing the configuration of the models
            to train.
        modality (str): Model type. One of [std, memory, encoder_memory] where
            std is the standard model, memory is the baseline that uses only the
            memory and encoder_memory is Memory Wrap
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device:{device}")

    modality='std'
    # get dataset info
    dataset_name = config['dataset_name']
    num_classes = config[dataset_name]['num_classes']
    config['dataset_dir'] = '../datasets/'

    # training parameters
    loss_criterion = torch.nn.NLLLoss()

    # saving/loading stuff
    save = config['save']
    path_saving_model = f"../models/{dataset_name}/proto/{config['model']}/{config['train_examples']}/"
    if save and not os.path.isdir(path_saving_model):
        os.makedirs(path_saving_model)

    # optimizer parameters
    learning_rate = float(config['optimizer']['learning_rate'])
    weight_decay = float(config['optimizer']['weight_decay'])
    nesterov = bool(config['optimizer']['nesterov'])
    momentum = float(config['optimizer']['momentum'])
    dict_optim = {'lr' :learning_rate, 'momentum':momentum,
                  'weight_decay':weight_decay, 'nesterov':nesterov}
    opt_milestones = config[dataset_name]['opt_milestones']

    run_acc = []
    initial_run = 0
    if FLAGS.continue_train:
        # load model
        print("Restarting training process\n")
        info = pickle.load( open(  path_saving_model+"conf.p", "rb" ) )
        initial_run = info['run_num']
        run_acc = info['accuracies']
    for run in range(initial_run,config['runs']):
        run_time = time.time()
        utils.set_seed(run)
        torch.cuda.init()
        model = utils.get_model(config['model'],num_classes,model_type=modality)
        model = model.to(device)
        model.linear = torch.nn.Identity()
        # training parameters
        if FLAGS.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(),**dict_optim)
        elif FLAGS.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(),
                                lr=float(config['optimizer']['learning_rate']))
        else:
            raise ValueError("Optimizer not supported")
        if dataset_name == 'CINIC10':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                T_max=config[dataset_name]['num_epochs'])
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,  milestones=opt_milestones)
        # get dataset
        train_loader, _, test_loader, mem_loader = utils.get_loaders(config,run,balanced=True)
        train_time = time.time()
        model = train_model(model, loaders= [train_loader,mem_loader], optimizer=optimizer,
                        scheduler=scheduler, loss_criterion=loss_criterion,
                        num_epochs=config[dataset_name]['num_epochs'], device=device)
        init_eval_time = time.time()
        best_acc, _ = eval_model(model,test_loader, mem_loader,loss_criterion,device)
        end_eval_time = time.time()

        # stats
        run_acc.append(best_acc)

        # save
        if save and path_saving_model:
            saved_name = f"{run+1}.pt"
            save_path = os.path.join(path_saving_model, saved_name)
            torch.save({'model_state_dict':model.state_dict(),
            'train_examples': config['train_examples'],
            'mem_examples':  config[config['dataset_name']]['mem_examples'],
            'model_name': config['model'],
            'num_classes': num_classes, 'modality':modality,
            'dataset_name':config['dataset_name']} , save_path)
            info = {'run_num':run+1,'accuracies':run_acc}
            pickle.dump( info, open( path_saving_model+"conf.p", "wb" ) )

        # log
        print(f"Run:{run+1} | Accuracy {best_acc:.2f} | " \
              f"Mean Accuracy:{np.mean(run_acc):.2f} | Std Dev Accuracy:{np.std(run_acc):.2f}\t" \
              f"T:{(train_time -run_time)/60:.2f}min\tE:{(end_eval_time -init_eval_time)/60:.2f}")

def main(argv):
    config_file = open(r'../config/train.yaml', 'r')
    config = yaml.safe_load(config_file)

    print(f"Model:{config['model']}\nSizeTrain:{config['train_examples']}\n")
    run_experiment(config)

if __name__ == '__main__':
    absl.app.run(main)
