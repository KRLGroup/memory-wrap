
import torch # type: ignore
import torchvision # type: ignore
import numpy
import random

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

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


def undo_normalization_SVHN(input):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    unnormalize = NormalizeInverse(mean,std)
    input = unnormalize(input)
    return input

def undo_normalization_CIFAR10(input):
    mean=[0.4914, 0.4822, 0.4465]
    std=[0.2023, 0.1994, 0.2010]
    unnormalize = NormalizeInverse(mean,std)
    input = unnormalize(input)
    return input



def split_dataset(dataset,train_size,val_size,seed):
    first_split = len(dataset) - val_size
    data_rest, val_dataset = torch.utils.data.random_split(dataset,[first_split,val_size],generator=torch.Generator().manual_seed(seed))
    size_train = min(len(data_rest),train_size)
    train_dataset, _ = torch.utils.data.random_split(data_rest,[size_train,len(data_rest)-size_train], generator=torch.Generator().manual_seed(seed))
    return train_dataset, val_dataset



def get_SVHN(data_dir, batch_size_train, batch_size_test,batch_size_memory,size_train=100000,seed=42):

    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            normalize,
    ])
    
    train_data = torchvision.datasets.SVHN(data_dir, split='train', download=True, transform=transforms)
    test_data =  torchvision.datasets.SVHN(data_dir, split='test', download=True, transform=transforms)
    train_dataset, val_dataset = split_dataset(train_data,size_train,6000,seed)


    train_loader = torch.utils.data.DataLoader( train_dataset, batch_size=batch_size_train,pin_memory=True, shuffle=True, drop_last=True,worker_init_fn=seed_worker)

    val_loader = torch.utils.data.DataLoader( val_dataset, batch_size=batch_size_test,pin_memory=True, shuffle=False,worker_init_fn=seed_worker)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_test,pin_memory=True, shuffle=False,worker_init_fn=seed_worker)

    mem_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_memory,pin_memory=True, shuffle=True, drop_last=True,worker_init_fn=seed_worker)
    
    return train_loader, val_loader, test_loader, mem_loader



def get_CIFAR10(data_dir, batch_size_train, batch_size_test,batch_size_memory,size_train=100000,seed=42):
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
    test_data = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=transforms_test)
    

    train_dataset, val_dataset = split_dataset(train_data,size_train,6000,seed)


    train_loader = torch.utils.data.DataLoader( train_dataset, batch_size=batch_size_train,pin_memory=True, shuffle=True, drop_last=True,worker_init_fn=seed_worker)

    val_loader = torch.utils.data.DataLoader( val_dataset, batch_size=batch_size_test,pin_memory=True,worker_init_fn=seed_worker)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_test,pin_memory=True,worker_init_fn=seed_worker)

    mem_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_memory,pin_memory=True, shuffle=True, drop_last=True,worker_init_fn=seed_worker)
    
    return train_loader, val_loader, test_loader, mem_loader

def get_CIFAR100(data_dir, batch_size_train, batch_size_test,batch_size_memory,size_train=100000,seed=42):
    normalize = torchvision.transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                 std=[0.2675, 0.2565, 0.2761])
    transforms_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
               torchvision.transforms.ToTensor(),
                normalize
    ])
    transforms_test = torchvision.transforms.Compose([
               torchvision.transforms.ToTensor(),
               normalize
    ])

    train_data = torchvision.datasets.CIFAR100(data_dir, train=True, download=True, transform=transforms_train)
    test_data = torchvision.datasets.CIFAR100(data_dir, train=False, download=True, transform=transforms_test)


    train_dataset, val_dataset = split_dataset(train_data,size_train,6000,seed)


    train_loader = torch.utils.data.DataLoader( train_dataset, batch_size=batch_size_train,pin_memory=True, shuffle=True, drop_last=True,worker_init_fn=seed_worker)

    val_loader = torch.utils.data.DataLoader( val_dataset, batch_size=batch_size_test,pin_memory=True,worker_init_fn=seed_worker)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_test,pin_memory=True,worker_init_fn=seed_worker)

    mem_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_memory,pin_memory=True, shuffle=True, drop_last=True,worker_init_fn=seed_worker)
    
    return train_loader, val_loader, test_loader, mem_loader

def get_CINIC10(data_dir, batch_size_train, batch_size_test,batch_size_memory,size_train=100000,seed=42):
    normalize = torchvision.transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404], std=[0.24205776, 0.23828046, 0.25874835])
  
    cinic_directory = data_dir+ "CINIC10/"

    data = torchvision.datasets.ImageFolder(cinic_directory + '/train',
    	transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
        normalize]))
    # get subset of train dataset
    train_dataset, _ = split_dataset(data,size_train,10,seed)
    train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size_train, shuffle=True)

    mem_loader = torch.utils.data.DataLoader(    train_dataset,   batch_size=batch_size_memory, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(cinic_directory + '/test',
    	transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
        normalize])),
    batch_size=batch_size_test, shuffle=False)

    val_loader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(cinic_directory + '/valid',
    	transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
       normalize])),
    batch_size=batch_size_test, shuffle=False)
    
    return train_loader, val_loader, test_loader, mem_loader