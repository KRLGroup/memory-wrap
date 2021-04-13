
import torch
import torch.nn as nn
import numpy as np
import random
import absl.flags
import absl.app
import os
import yaml
import datasets
from aux import get_model,eval_memory,get_loaders,eval_std, eval_memory_vote
import time
import pickle

# user flags
absl.flags.DEFINE_string("modality", None, "std, memory or encoder_memory")
absl.flags.DEFINE_bool("continue_train", False, "std, memory or mlp")
absl.flags.DEFINE_integer("log_interval",100,"Log interval between prints during training process")
absl.flags.mark_flag_as_required("modality")
FLAGS = absl.flags.FLAGS

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # set to false for reproducibility, True to boost performance
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    random_state = random.getstate()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return random_state

def train_step(model, inputs, targets, optimizer, loss_criterion):    
    optimizer.zero_grad() # zeroing gradients
    if type(inputs) is tuple:
        outputs  = model(*inputs) # get output
    else:
        outputs = model(inputs)
    loss = loss_criterion(outputs, targets)  # compute loss
    loss.backward() # compute gradients
    optimizer.step() # update weights
    return loss

def train_memory_model(model,loaders,optimizer,scheduler, loss_criterion, num_epochs,device):
        
    train_loader, val_loader, mem_loader = loaders
    
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
            #train_step(model=model,inputs=(data,memory_input),targets=y,optimizer=optimizer,loss_criterion=loss_criterion)
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



def train_std_model(model,train_loader,optimizer,scheduler, loss_criterion, num_epochs, device=torch.device('cpu')):
        
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
            #train_step(model=model,inputs=data,targets=y,optimizer=optimizer,loss_criterion=loss_criterion)
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



def run_experiment(config,modality):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:{}".format(device))

    # get dataset info
    dataset_name = config['dataset_name']
    num_classes = config[dataset_name]['num_classes']


    # training parameters
    loss_criterion = torch.nn.CrossEntropyLoss()

    # saving/loading stuff
    save = False #config['save']
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

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
       
    for run in range(initial_run,config['runs']):
        run_time = time.time()
        set_seed(run)
        torch.cuda.init()
        model = get_model(config['model'],num_classes,model_type=modality)
        model = model.to(device)
        # training parameters
        optimizer = torch.optim.SGD(model.parameters(),**dict_optim)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,  milestones=opt_milestones)
        
        # get dataset
        train_loader, val_loader, test_loader, mem_loader = get_loaders(config,run)

         # training process
        if modality == 'memory' or modality == 'encoder_memory':
            model = train_memory_model(model,[train_loader,val_loader,mem_loader],optimizer,scheduler,loss_criterion,config[dataset_name]['num_epochs'],device=device)  
            train_time = time.time()

            cum_acc =  []

            # perform 10 times the validation to stabilize results (due to random selection of memory samples)
            init_eval_time = time.time()
            for _ in range(5):
                best_acc, best_loss = eval_memory(model,test_loader, mem_loader,loss_criterion,device)
                cum_acc.append(best_acc)
            best_acc = np.mean(cum_acc)
            end_eval_time = time.time()

        else:
            model = train_std_model(model,train_loader,optimizer,scheduler,loss_criterion,config[dataset_name]['num_epochs'],device)
            train_time = time.time()
            init_eval_time = time.time()
            best_acc, best_loss  = eval_std(model,test_loader,loss_criterion,device)
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
            'num_classes': num_classes} , save_path)
            info = {'run_num':run+1,
                'accuracies': run_acc}
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