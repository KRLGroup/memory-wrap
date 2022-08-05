
import torch # type: ignore
import numpy as np
import absl.flags
import absl.app
import os
import yaml
import utils.utils as utils
import time
import pickle
from typing import List

# user flags
absl.flags.DEFINE_string("modality", None, "std, memory or encoder_memory")
absl.flags.DEFINE_bool("continue_train", False, "std, memory or mlp")
absl.flags.DEFINE_integer("log_interval",100,"Log interval between prints during training process")
absl.flags.mark_flag_as_required("modality")
FLAGS = absl.flags.FLAGS

def train_memory_model(model:torch.nn.Module,loaders:List[torch.utils.data.DataLoader],optimizer:torch.optim.Optimizer,scheduler:torch.optim.lr_scheduler._LRScheduler,loss_criterion:torch.nn.modules.loss, num_epochs:int,device:torch.device)->torch.nn.Module:
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
        for batch_idx, (data, y) in enumerate(train_loader):
            
            optimizer.zero_grad()
            # input
            data = data.to(device)
            y = y.to(device)
            memory_input, _ = next(iter(mem_loader))
            memory_input = memory_input.to(device)
            
            # perform training step
            with torch.cuda.amp.autocast():
                outputs  = model(data,memory_input)
                loss = loss_criterion(outputs, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            #log stuff
            if batch_idx % FLAGS.log_interval == 0:
                print('Train Epoch: {} [({:.0f}%({})]\t'.format(
                epoch,
                100. * batch_idx / len(train_loader), len(train_loader.dataset)),end='\r')

        scheduler.step()# increase scheduler step for each epoch

    return model

def train_std_model(model:torch.nn.Module,train_loader:torch.utils.data.DataLoader,optimizer:torch.optim.Optimizer,scheduler:torch.optim.lr_scheduler._LRScheduler, loss_criterion:torch.nn.modules.loss, num_epochs:int, device:torch.device=torch.device('cpu'))->torch.nn.Module:
    """ Function to train standard models

    Args:
        model (torch.nn.Module): standard PyTorch model
        train_loader (torch.utils.data.DataLoader): training dataset
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
    # training process
    model.train()  
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(1, num_epochs + 1):      
        for batch_idx, (data, y) in enumerate(train_loader): 
            optimizer.zero_grad() 
            # input
            data = data.to(device)
            y = y.to(device)
            
            # training step
            with torch.cuda.amp.autocast():
                outputs  = model(data)
                loss = loss_criterion(outputs, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # log stuff
            if batch_idx % FLAGS.log_interval == 0:
                print('Train Epoch: {} [({:.0f}%({})]\t'.format(
                epoch,
                100. * batch_idx / len(train_loader), len(train_loader.dataset)),end='\r')
        
        scheduler.step() # increase scheduler step for each epoch

    return model

def run_experiment(config:dict,modality:str):
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
    print("Device:{}".format(device))

    # get dataset info
    dataset_name = config['dataset_name']
    num_classes = config[dataset_name]['num_classes']


    # training parameters
    loss_criterion = torch.nn.CrossEntropyLoss()

    # saving/loading stuff
    save = config['save']
    path_saving_model = 'models/{}/{}/{}/{}/'.format(dataset_name,FLAGS.modality, config['model'],config['train_examples'])
    if save and not os.path.isdir(path_saving_model): 
        os.makedirs(path_saving_model)
    
    # optimizer parameters
    learning_rate = float(config['optimizer']['learning_rate'])
    weight_decay = float(config['optimizer']['weight_decay'])
    nesterov = bool(config['optimizer']['nesterov'])
    momentum = float(config['optimizer']['momentum'])
    dict_optim = {'lr' :learning_rate, 'momentum':momentum, 'weight_decay':weight_decay, 'nesterov':nesterov}
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
        # training parameters
        optimizer = torch.optim.SGD(model.parameters(),**dict_optim)
        if dataset_name == 'CINIC10':
             scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config[dataset_name]['num_epochs'])
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,  milestones=opt_milestones)
        # get dataset
        train_loader, _, test_loader, mem_loader = utils.get_loaders(config,run)

         # training process
        if modality == 'memory' or modality == 'encoder_memory':
            model = train_memory_model(model,[train_loader,mem_loader],optimizer,scheduler,loss_criterion,config[dataset_name]['num_epochs'],device=device)  
            train_time = time.time()

            cum_acc =  []

            # perform 5 times the validation to stabilize results (due to random selection of memory samples)
            init_eval_time = time.time()
            for _ in range(5):
                best_acc, best_loss = utils.eval_memory(model,test_loader, mem_loader,loss_criterion,device)
                cum_acc.append(best_acc)
            best_acc = np.mean(cum_acc)
            end_eval_time = time.time()

        else:
            model = train_std_model(model,train_loader,optimizer,scheduler,loss_criterion,config[dataset_name]['num_epochs'],device)
            train_time = time.time()
            init_eval_time = time.time()
            best_acc, best_loss  = utils.eval_std(model,test_loader,loss_criterion,device)
            end_eval_time = time.time()

        # stats
        run_acc.append(best_acc)

        # save
        if save and path_saving_model:
            saved_name = "{}.pt".format(run+1)
            save_path = os.path.join(path_saving_model, saved_name)
            torch.save({'model_state_dict':model.state_dict(),
            'train_examples': config['train_examples'],
            'mem_examples':  config[config['dataset_name']]['mem_examples'],
            'model_name': config['model'],
            'num_classes': num_classes, 'modality':modality, 'dataset_name':config['dataset_name']} , save_path)
            info = {'run_num':run+1,}
            pickle.dump( info, open( path_saving_model+"conf.p", "wb" ) )

        # log
        print("Run:{} | Best Loss:{:.4f} | Accuracy {:.2f} | Last Loss: Accuracy:| Mean Accuracy:{:.2f} | Std Dev Accuracy:{:.2f}\tT:{:.2f}min\tE:{:.2f}".format(run+1,best_loss,best_acc, np.mean(run_acc), np.std(run_acc),(train_time -run_time)/60,(end_eval_time -init_eval_time)/60))







def main(argv):

    config_file = open(r'config/train.yaml')
    config = yaml.safe_load(config_file)

    print("Model:{}\nSizeTrain:{}\n".format(config['model'], config['train_examples']))
    run_experiment(config, FLAGS.modality)

if __name__ == '__main__':
  absl.app.run(main)