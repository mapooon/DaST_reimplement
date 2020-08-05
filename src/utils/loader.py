import torch
from torchvision import datasets,transforms

def get_loader(name):
    assert name in ['mnist','cifar']

    if name == 'mnist':
        my_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)
        ])
        trainval_dataset=datasets.MNIST("data",train=True,download=True,transform=my_transform)
        train_size=int(len(trainval_dataset)*0.8)
        val_size=len(trainval_dataset)-train_size
        train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])
        train_loader=torch.utils.data.DataLoader(
            train_dataset,
            batch_size=128,
            shuffle=True
        )
        val_dataset=torch.utils.data.DataLoader(
            val_dataset,
            batch_size=128,
            shuffle=True
        )
        test_loader=torch.utils.data.DataLoader(
            datasets.MNIST("data",train=False,download=True,transform=my_transform),
            batch_size=1,
            shuffle=False
        )
    else:
        mean_cifar10 = [0.485, 0.456, 0.406]   
        std_cifar10 = [0.229, 0.224, 0.225]
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean_cifar10, std_cifar10)])
        trainval_dataset=datasets.CIFAR10("data",train=True,download=True,transform=transform)
        train_size=int(len(trainval_dataset)*0.8)
        val_size=len(trainval_dataset)-train_size
        train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])
        train_loader=torch.utils.data.DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=True
            )
        val_loader=torch.utils.data.DataLoader(
            val_dataset,
            batch_size=64,
            shuffle=True
            )
        test_loader=torch.utils.data.DataLoader(
            datasets.CIFAR10("data",train=False,download=True,transform=transform),
            batch_size=1,
            shuffle=False
            )

    dataloaders={
                'train':train_loader,
                'val':val_loader,
                'test':test_loader
                }
    
    return dataloaders

    