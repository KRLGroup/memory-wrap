import datasets
import torch
import torch.nn as nn
from architectures.resnet import ResNet18, MemoryResNet18, EncoderMemoryResNet18
from architectures.mobilenet import MobileNetV2, MemoryMobileNetV2,EncoderMemoryMobileNetV2
from architectures.efficientnet import EfficientNetB0, MemoryEfficientNetB0, EncoderMemoryEfficientNetB0

import statistics
def get_model(model_name, num_classes, model_type='memory'):
    if model_name == 'efficientnet':
        if model_type=='memory':
            model = MemoryEfficientNetB0(num_classes)
        elif model_type == 'encoder_memory':
            model = EncoderMemoryEfficientNetB0(num_classes)
        else:
            model = EfficientNetB0(num_classes)
    elif model_name == 'resnet18':
        if model_type=='memory':
            model = MemoryResNet18(num_classes)
        elif model_type == 'encoder_memory':
            model = EncoderMemoryResNet18(num_classes)
        else:
            model = ResNet18(num_classes)

    elif model_name == 'mobilenet':
        if model_type=='memory':
            model = MemoryMobileNetV2(num_classes)
        elif model_type == 'encoder_memory':
            model = EncoderMemoryMobileNetV2(num_classes)
        else:
            model = MobileNetV2(num_classes)
    else:
        print("Error: input model name is not valid!")
        exit()

   
    return model

def get_loaders(config,seed=42):
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
    loaders = load_dataset(data_dir,batch_size_train=batch_size_train, batch_size_test=batch_size_test,batch_size_memory=mem_examples,size_train=train_examples,seed=seed)
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
            memory, _ = next(iter(mem_loader))
            memory = memory.to(device)

            output  = model(data,memory)
            loss = loss_criterion(output, target) 
            pred = output.data.max(1, keepdim=True)[1]
            
            correct += pred.eq(target.data.view_as(pred)).sum().item()
            test_loss += loss.item()
            
        
        test_accuracy = 100.*(torch.true_divide(correct,len(loader.dataset)))
    return test_accuracy,  test_loss

def eval_memory_vote(model,loader,mem_loader,loss_criterion,device):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for _, (data, target) in enumerate(loader):
            data = data.to(device)
            target = target.to(device)
            outputs = []
            for _ in range(10):
                memory, _ = next(iter(mem_loader))
                memory = memory.to(device)

                output  = model(data,memory)
                pred = output.data.max(1, keepdim=True)[1]
                outputs.append(pred)
            prediction = statistics.mode(outputs)

            loss = loss_criterion(output, target) 
            
            correct += prediction.eq(target.data.view_as(prediction)).sum().item()
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
    return test_accuracy, test_loss