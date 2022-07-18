
import torch # type: ignore
import torchvision # type: ignore
import numpy
import random
from typing import List

def seed_worker(worker_id:int):
    """Function to set the seed of workers

    Args:
        worker_id (int): worker id
    """

    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)   
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight

class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


def undo_normalization_SVHN(input:torch.Tensor)->torch.Tensor:
    """Function to revert the normalization of an input image

    Args:
        input (torch.Tensor): input image

    Returns:
        torch.Tensor: Input image where normalization has been removed
    """
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    unnormalize = NormalizeInverse(mean,std)
    input = unnormalize(input)
    return input

def undo_normalization_CIFAR10(input:torch.Tensor)->torch.Tensor:
    """Function to revert the normalization of an input image

    Args:
        input (torch.Tensor): input image

    Returns:
        torch.Tensor: Input image where normalization has been removed
    """
    mean=[0.4914, 0.4822, 0.4465]
    std=[0.2023, 0.1994, 0.2010]
    unnormalize = NormalizeInverse(mean,std)
    input = unnormalize(input)
    return input

def undo_normalization_CINIC10(input:torch.Tensor)->torch.Tensor:
    """Function to revert the normalization of an input image

    Args:
        input (torch.Tensor): input image

    Returns:
        torch.Tensor: Input image where normalization has been removed
    """
    mean = [0.47889522, 0.47227842, 0.43047404]
    std=[0.24205776, 0.23828046, 0.25874835]
    unnormalize = NormalizeInverse(mean,std)
    input = unnormalize(input)
    return input


def split_dataset(dataset:torch.utils.data.Dataset,train_size:int,val_size:int,seed:int)->List[torch.utils.data.Dataset]:
    """ Function to split a dataset
    Args:
        dataset (torch.utils.data.Dataset): dataset to be splitted
        train_size (int): Number of samples to keep in training dataset
        val_size (int): Number of samples to keep in validation dataset
        seed (int): Seed to fix split for reproducibility
    Returns:
        List[torch.utils.data.Dataset]: The subsets of dataset chosen for
            training dataset and validation dataset 
    """
    first_split = len(dataset) - val_size
    data_rest, val_dataset = torch.utils.data.random_split(dataset,[first_split,val_size],generator=torch.Generator().manual_seed(seed))
    size_train = min(len(data_rest),train_size)
    train_dataset, _ = torch.utils.data.random_split(data_rest,[size_train,len(data_rest)-size_train], generator=torch.Generator().manual_seed(seed))
    return train_dataset, val_dataset



def get_SVHN(data_dir:str, batch_size_train:int, batch_size_test:int, batch_size_memory:int,
             size_train:int=100000, balanced:bool=False, seed:int=42) -> List[torch.utils.data.DataLoader]:
    """ Function to retrieve SVHN dataset and dataloader

    Args:
        data_dir (str): path where the dataset is stored
        batch_size_train (int): Batch size to be used for training dataset
        batch_size_test (int): Batch size to be used for testing dataset
        batch_size_memory (int): Number of samples in memory set
        size_train (int, optional): Number of samples in the whole training
             dataset. Defaults to 100000.
        seed (int, optional): Seed to ensure reproducibility. Defaults to 42.

    Returns:
        List[torch.utils.data.DataLoader]: Dataloaders for training, validation,
        testing dataset and for memory sets
    """

    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            normalize,
    ])
    
    train_data = torchvision.datasets.SVHN(data_dir, split='train', download=True, transform=transforms)
    test_data =  torchvision.datasets.SVHN(data_dir, split='test', download=True, transform=transforms)
    train_dataset, val_dataset = split_dataset(train_data,size_train,6000,seed)

    if balanced:
        # https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
        weights = make_weights_for_balanced_classes(train_dataset, 10)                                                                
        weights = torch.DoubleTensor(weights) 
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights),replacement=False)                     
        mem_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_memory, sampler = sampler, drop_last=True, pin_memory=True)     
    else:
        mem_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_memory,pin_memory=True, shuffle=True, drop_last=True,worker_init_fn=seed_worker)
    train_loader = torch.utils.data.DataLoader( train_dataset, batch_size=batch_size_train,pin_memory=True, shuffle=True, drop_last=True,worker_init_fn=seed_worker)

    val_loader = torch.utils.data.DataLoader( val_dataset, batch_size=batch_size_test,pin_memory=True, shuffle=False,worker_init_fn=seed_worker)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_test,pin_memory=True, shuffle=False,worker_init_fn=seed_worker)

    return train_loader,val_loader, test_loader, mem_loader


def get_SVHN_dataset(data_dir:str, size_train:int=100000,seed:int=42)->List[torch.utils.data.DataLoader]:
    """ Function to retrieve SVHN dataset and dataloader

    Args:
        data_dir (str): path where the dataset is stored
        batch_size_train (int): Batch size to be used for training dataset
        batch_size_test (int): Batch size to be used for testing dataset
        batch_size_memory (int): Number of samples in memory set
        size_train (int, optional): Number of samples in the whole training
             dataset. Defaults to 100000.
        seed (int, optional): Seed to ensure reproducibility. Defaults to 42.

    Returns:
        List[torch.utils.data.DataLoader]: Dataloaders for training, validation,
        testing dataset and for memory sets
    """

    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            normalize,
    ])
    
    train_data = torchvision.datasets.SVHN(data_dir, split='train', download=True, transform=transforms)
    test_data =  torchvision.datasets.SVHN(data_dir, split='test', download=True, transform=transforms)
    train_dataset, val_dataset = split_dataset(train_data,size_train,6000,seed)   
    return train_dataset,val_dataset, test_data


def get_CIFAR10(data_dir:str, batch_size_train:int, batch_size_test:int,batch_size_memory:int,size_train:int=100000,seed:int=42)->List[torch.utils.data.DataLoader]:
    """ Function to retrieve CIFAR10 dataset and dataloader

    Args:
        data_dir (str): path where the dataset is stored
        batch_size_train (int): Batch size to be used for training dataset
        batch_size_test (int): Batch size to be used for testing dataset
        batch_size_memory (int): Number of samples in memory set
        size_train (int, optional): Number of samples in the whole training
             dataset. Defaults to 100000.
        seed (int, optional): Seed to ensure reproducibility. Defaults to 42.

    Returns:
        List[torch.utils.data.DataLoader]: Dataloaders for training, validation,
        testing dataset and for memory sets
    """
    normalize = torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])
    transforms_train = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
               torchvision.transforms.ToTensor(),
                normalize
    ])
    transforms_test = torchvision.transforms.Compose([
               torchvision.transforms.ToTensor(),
               normalize
    ])

    train_data = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=transforms_train)
    memory_data = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=transforms_test)
    test_data = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=transforms_test)
    

    train_dataset, val_dataset = split_dataset(train_data,size_train,6000,seed)


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train,pin_memory=True, shuffle=True, drop_last=True,worker_init_fn=seed_worker)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_test,pin_memory=True,worker_init_fn=seed_worker)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_test,pin_memory=True,worker_init_fn=seed_worker)

    mem_loader = torch.utils.data.DataLoader(memory_data, batch_size=batch_size_memory,pin_memory=True, shuffle=True, drop_last=True,worker_init_fn=seed_worker)
    
    return train_loader, val_loader, test_loader, mem_loader


def get_CINIC10(data_dir:str, batch_size_train:int, batch_size_test:int,batch_size_memory:int,size_train:int=100000,seed:int=42)->List[torch.utils.data.DataLoader]:
    """ Function to retrieve CINIC10 dataset and dataloader

    Args:
        data_dir (str): path where the dataset is stored
        batch_size_train (int): Batch size to be used for training dataset
        batch_size_test (int): Batch size to be used for testing dataset
        batch_size_memory (int): Number of samples in memory set
        size_train (int, optional): Number of samples in the whole training
             dataset. Defaults to 100000.
        seed (int, optional): Seed to ensure reproducibility. Defaults to 42.

    Returns:
        List[torch.utils.data.DataLoader]: Dataloaders for training, validation,
        testing dataset and for memory sets
    """
    normalize = torchvision.transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404], std=[0.24205776, 0.23828046, 0.25874835])
  
    cinic_directory = data_dir+ "/CINIC10"
    data = torchvision.datasets.ImageFolder(cinic_directory + '/train',
    	transform=torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
        normalize]))
    # get subset of train dataset
    train_dataset, _ = split_dataset(data,size_train,10,seed)
    train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size_train, drop_last=True, shuffle=True)

    mem_loader = torch.utils.data.DataLoader(train_dataset, 
                 batch_size=batch_size_memory, drop_last=True,
                 shuffle=True)

    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(cinic_directory + '/test',
    	transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
        normalize])),
    batch_size=batch_size_test, shuffle=True)

    val_loader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(cinic_directory + '/valid',
    	transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
       normalize])),
    batch_size=batch_size_test, shuffle=False)
    
    return train_loader, val_loader, test_loader, mem_loader