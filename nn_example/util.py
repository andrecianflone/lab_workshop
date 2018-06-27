import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

def get_data(batch_size):
    train_dataset = dset.MNIST(root='../data',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

    test_dataset = dset.MNIST(root='../data',
                         train=False,
                         transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    return train_loader, test_loader

