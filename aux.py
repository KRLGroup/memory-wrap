import datasets
import torch
import torch.nn as nn
from architectures import resnet
from architectures.mobilenet import MobileNetV2
from architectures.efficientnet import EfficientNetB0
from architectures.memory import MemoryWrap,EncoderMemoryWrap
from architectures.memory import Identity

def get_model(model_name, num_classes, model_type='memory'):
    if model_name == 'efficientnet':
        model = EfficientNetB0()
        num_input_feat = 320
    elif model_name == 'resnet18':
        model = resnet.ResNet18()
        num_input_feat = 512
    elif model_name == 'mobilenet':
        model = MobileNetV2()
        num_input_feat = 1280
    else:
        print("Error: input model name is not valid!")
        exit()
    if model_type=='memory':
        # replace last layer of a given architecture. It avoid to re-write all architectures separately
        if model_name == 'resnet18':
            model.fc = Identity()
        else:
            model.linear = Identity()
        # add memory wrap to the model
        model = MemoryWrap(model,num_input_feat,num_classes)
    elif model_type=='encoder_memory':
        # replace last layer of a given architecture. It avoid to re-write all architectures separately
        if model_name == 'resnet18':
            model.fc = Identity()
        else:
            model.linear = Identity()
        # add memory wrap to the model
        model = EncoderMemoryWrap(model,num_input_feat,num_classes)
    else:
        if model_name == 'resnet18':
            model.fc = nn.Linear(num_input_feat,num_classes)
        else:
            model.linear = nn.Linear(num_input_feat,num_classes)

    
    return model

def get_loaders(config):
    # unpack config
    dataset = config['dataset_name']
    data_dir = config['dataset_dir']
    batch_size_train = config['batch_size_train']
    batch_size_test = config['batch_size_test']
    train_examples = config['train_examples']
    mem_examples = config[dataset]['mem_examples']
    assert train_examples > batch_size_train

    #load data
    load_dataset = getattr(datasets, 'get_'+dataset)
    loaders = load_dataset(data_dir,batch_size_train=batch_size_train, batch_size_test=batch_size_test,batch_size_memory=mem_examples,size_train=train_examples)
    return loaders

def get_datasets(config):
    dataset = config['dataset_name']
    data_dir = config['dataset_dir']
    train_examples = config['train_examples']
    load_dataset = getattr(datasets, 'get_'+dataset)
    return load_dataset(data_dir,size_train=train_examples)

def eval_memory(model,loader,mem_loader,loss_criterion,device):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for _, (data, target) in enumerate(loader):
            data = data.to(device)
            target = target.to(device)
            memory, y = next(iter(mem_loader))
            memory = memory.to(device)

            output  = model(data,memory)
            loss = loss_criterion(output, target) 
            pred = output.data.max(1, keepdim=True)[1]
            
            correct += pred.eq(target.data.view_as(pred)).sum().item()
            test_loss += loss.item()
            
        
        test_accuracy = 100.*(torch.true_divide(correct,len(loader.dataset)))
    return test_accuracy,  torch.true_divide(test_loss,len(loader))

def eval_std(model,loader,loss_criterion,device):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for _, (data, target) in enumerate(loader):
            data = data.to(device)
            target = target.to(device)
            
            output = model(data)

            loss = loss_criterion(output, target) 
            pred = output.data.max(1, keepdim=True)[1]
            
            correct += pred.eq(target.data.view_as(pred)).sum().item()
            test_loss += loss.item()
            
        test_accuracy = 100.*(torch.true_divide(correct,len(loader.dataset)))
    return test_accuracy, torch.true_divide(test_loss,len(loader))