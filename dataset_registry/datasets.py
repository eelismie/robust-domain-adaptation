from .mnistm import MNISTM
from torch.utils.data import DataLoader
from torchvision import datasets
import torch
import torchvision.transforms as transforms
import os

class MNIST():

    def __init__(self,path,opt = {}):
        self.path = path 
        self.opt = opt
        
    def get_loaders(self):

        opt = self.opt

        os.makedirs(self.path, exist_ok=True)

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                self.path,
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(opt["img_size"]), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ),
            ),
            batch_size=opt["batch_size"],
            shuffle=True,
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                self.path,
                train=False,
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(opt["img_size"]),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5])
                    ]
                ),
            ),
            batch_size=opt["batch_size"],
            shuffle=True,
        )

        return train_loader, test_loader


class MNIST_M():

    def __init__(self,path,opt = {}):
        self.path = path 
        self.opt = opt
        
    def get_loaders(self):

        opt = self.opt

        os.makedirs(self.path, exist_ok=True)
        train_loader = torch.utils.data.DataLoader(
            MNISTM(
                "data/mnistm",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.Resize(opt["img_size"]),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                ),
            ),
            batch_size=opt["batch_size"],
            shuffle=True,
        )

        test_loader = torch.utils.data.DataLoader(
            MNISTM(
                "data/mnistm",
                train=False,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.Resize(opt["img_size"]),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                ),
            ),
            batch_size=opt["batch_size"],
            shuffle=True,
        )

        return train_loader, test_loader
