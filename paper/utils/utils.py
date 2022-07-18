from typing import List, Tuple
import os 
import random

import torch # type: ignore
import numpy as np
import torch.nn as nn
from numpy.random import RandomState

import utils.datasets as datasets
from architectures import resnet
from architectures import mobilenet
from architectures import efficientnet
from architectures import shufflenet
from architectures import densenet
from architectures import googlenet 
from architectures import wide_resnet


_EPSILON = 1e-6


def vector_distance(a:torch.Tensor , b:torch.Tensor, distance_type:str='cosine')->torch.Tensor:
    """
    Compute the distance between two vectors.
    Args:
        a: first vector
        b: second vector
        distance_type: type of distance to use. Can be 'cosine', 'l2' or 'dot'
    Returns:
        distance: distance between x and y
    """

    def _vector_norms(vector:torch.Tensor)->torch.Tensor:
        """ Computes the vector norms
        Args:
            vector: The vector from which there must be calculated the norms

        Returns:
                A tensor containing the norms of input vector v
        """

        squared_norms = torch.sum(vector * vector, dim=1, keepdim=True)
        return torch.sqrt(squared_norms + _EPSILON)

    if distance_type == 'cosine':
        a_norm = a / _vector_norms(a)
        b_norm = b / _vector_norms(b)
        distance = 1 - torch.mm(a_norm,b_norm.transpose(0,1))
    elif distance_type == 'l2':
        distance = (
            a.unsqueeze(1).expand(a.shape[0], b.shape[0], -1) -
            b.unsqueeze(0).expand(a.shape[0], b.shape[0], -1)
    ).pow(2).sum(dim=2)
    elif distance_type == 'dot':
        expanded_a = a.unsqueeze(1).expand(a.shape[0], b.shape[0], -1)
        expanded_b = b.unsqueeze(0).expand(a.shape[0], b.shape[0], -1)
        distance = - (expanded_a * expanded_b).sum(dim=2)
    return distance


def set_seed(seed: int) -> RandomState:
    """ Method to set seed across runs to ensure reproducibility.
    It fixes seed for single-gpu machines.

    Args:
        seed (int): Seed to fix reproducibility. It should different for
            each run 

    Returns:
        RandomState: fixed random state to initialize dataset iterators
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # set to false for reproducibility, True to boost performance
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    random_state = random.getstate()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return random_state

def get_model(model_name: str, num_classes:int , model_type:str) -> torch.nn.Module:
    """ Utility function to retrieve the requested model.

    Args:
        model_name (str): One of ['resnet18, efficientnet , mobilenet,
            shufflenet, googlenet, densenet]
        num_classes (int): Number of output units.
        model_type (str): Specify the model variant. 'std' is the standard
            model, 'memory' is the baseline that uses only the memory and
            'encoder_memory' is Memory Wrap

    Raises:
        ValueError: if model_name is not one of ['resnet18, efficientnet ,
            mobilenet, shufflenet, googlenet, densenet]
        ValueError: if model_type is not one of ['std','memory',                '   encoder_memory']

    Returns:
        torch.nn.Module: PyTorch model
    """
    if model_type not in ['memory','encoder_memory','std']:
        raise ValueError(f'modality (model type) must be one of [\'memory\',\'encoder_memory\',\'std\'], not {model_type}.')
    if model_name == 'efficientnet':
        if model_type=='memory':
            model = efficientnet.MemoryEfficientNetB0(num_classes)
        elif model_type == 'encoder_memory':
            model = efficientnet.EncoderMemoryEfficientNetB0(num_classes)
        else:
            model = efficientnet.EfficientNetB0(num_classes)
    elif model_name == 'wide-resnet':
        if model_type=='memory':
            model = wide_resnet.memory_wrn28_10()
        elif model_type == 'encoder_memory':
             model = wide_resnet.encoder_wrn28_10()
        else:
            model = wide_resnet.wrn28_10()
    elif model_name == 'resnet18':
        if model_type=='memory':
            model = resnet.MemoryResNet18(num_classes)
        elif model_type == 'encoder_memory':
            model = resnet.EncoderMemoryResNet18(num_classes)
        else:
            model = resnet.ResNet18(num_classes)
    elif model_name == 'resnet34':
            model = resnet.ResNet34(num_classes)
    elif model_name == 'densenet169':
            model = densenet.densenet_cifar_169()
    elif model_name == 'shufflenet1x':
             model = shufflenet.ShuffleNetV2(net_size=1)
    elif model_name == 'shufflenet':
        if model_type=='memory':
            model = shufflenet.MemoryShuffleNetV2(net_size=0.5,num_classes=num_classes)
        elif model_type == 'encoder_memory':
            model = shufflenet.EncoderMemoryShuffleNetV2(net_size=0.5,num_classes=num_classes)
        else:
            model = shufflenet.ShuffleNetV2(net_size=0.5,num_classes=num_classes)
    elif model_name == 'densenet':
        if model_type=='memory':
            model = densenet.memory_densenet_cifar()
        elif model_type == 'encoder_memory':
            model = densenet.encoder_memory_densenet_cifar()
        else:
            model = densenet.densenet_cifar()
    elif model_name == 'googlenet':
        if model_type=='memory':
            model = googlenet.MemoryGoogLeNet()
        elif model_type == 'encoder_memory':
            model = googlenet.EncoderMemoryGoogLeNet()
        else:
            model = googlenet.GoogLeNet()
    elif model_name == 'mobilenet':
        if model_type=='memory':
            model = mobilenet.MemoryMobileNetV2(num_classes)
        elif model_type == 'encoder_memory':
            model = mobilenet.EncoderMemoryMobileNetV2(num_classes)
        else:
            model = mobilenet.MobileNetV2(num_classes)
    else:
        raise ValueError("Error: input model name is not valid!")

   
    return model

def get_loaders(config: dict, seed: int=42, balanced:bool=False)-> List[torch.utils.data.DataLoader]:
    """ Retrieve the loaders (train, test, validation and memory) for
        the given dataset

    Args:
        config (dict): It is a dictionary containing the following keys
            ['dataset_name','dataset_dir','batch_size_train','batch_size_test',
            'train_examples',dataset_name->'mem_examples']

        seed (int, optional): Seed to ensure reproducibility. Defaults to 42.

    Returns:
        List[torch.utils.data.DataLoader]: List of loaders [train_loader, val_loader, test_loader, mem_loader]
    """
    # unpack config
    dataset = config['dataset_name']
    data_dir = config['dataset_dir']
    batch_size_train = config['batch_size_train']
    batch_size_test = config['batch_size_test']
    train_examples = config['train_examples']
    mem_examples = config[dataset]['mem_examples']

    if train_examples < batch_size_train:
        batch_size_train = train_examples

    #load data
    load_dataset = getattr(datasets, 'get_'+dataset, balanced)
    loaders = load_dataset(data_dir,batch_size_train=batch_size_train, batch_size_test=batch_size_test,batch_size_memory=mem_examples,size_train=train_examples,seed=seed)
    return loaders


def eval_memory(model:torch.nn.Module,loader:torch.utils.data.DataLoader,mem_loader:torch.utils.data.DataLoader,loss_criterion:torch.nn.modules.loss,device:torch.device)-> Tuple[float, float]:
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
            memory, _ = next(iter(mem_loader))
            memory = memory.to(device)

            output  = model(data,memory)
            loss = loss_criterion(output, target) 
            pred = output.data.max(1, keepdim=True)[1]
            
            correct += pred.eq(target.data.view_as(pred)).sum().item()
            test_loss += loss.item()
            
        
        test_accuracy = 100.*(torch.true_divide(correct,len(loader.dataset)))
    return test_accuracy,  test_loss



def eval_std(model:torch.nn.Module,loader:torch.utils.data.DataLoader,loss_criterion:torch.nn.modules.loss,device:torch.device)-> Tuple[float, float]:
    """ Method to evaluate a standrdpytorch model. The model takes an
        image as input and return the logits.

    Args:
        model (torch.nn.Module): Trained model to evaluate
        loader (torch.utils.data.DataLoader): loader containing testing
            samples where evaluate the model
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
            
            output = model(data)

            loss = loss_criterion(output, target) 
            pred = output.data.max(1, keepdim=True)[1]
            
            correct += pred.eq(target.data.view_as(pred)).sum().item()
            test_loss += loss.item()
            
        test_accuracy = 100.*(torch.true_divide(correct,len(loader.dataset)))
    return test_accuracy, test_loss