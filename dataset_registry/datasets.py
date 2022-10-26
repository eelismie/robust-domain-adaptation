import os
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms as transforms

from .mnistm import MNISTM
from torchvision import datasets
from torch.utils.data import DataLoader
from tllib.vision.transforms import ResizeImage
from tllib.vision.datasets.visda2017 import VisDA2017
from timm.data.auto_augment import auto_augment_transform, rand_augment_transform

def get_train_transform(resizing='default', scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), random_horizontal_flip=True,
                        random_color_jitter=False, resize_size=224, norm_mean=(0.485, 0.456, 0.406),
                        norm_std=(0.229, 0.224, 0.225), auto_augment=None):
    """
    resizing mode:
        - default: resize the image to 256 and take a random resized crop of size 224;
        - cen.crop: resize the image to 256 and take the center crop of size 224;
        - res: resize the image to 224;
    """
    transformed_img_size = 224
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.RandomResizedCrop(224, scale=scale, ratio=ratio)
        ])
    elif resizing == 'cen.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224)
        ])
    elif resizing == 'ran.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.RandomCrop(224)
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
        transformed_img_size = resize_size
    else:
        raise NotImplementedError(resizing)
    transforms = [transform]
    if random_horizontal_flip:
        transforms.append(T.RandomHorizontalFlip())
    if auto_augment:
        aa_params = dict(
            translate_const=int(transformed_img_size * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in norm_mean]),
            interpolation=Image.BILINEAR
        )
        if auto_augment.startswith('rand'):
            transforms.append(rand_augment_transform(auto_augment, aa_params))
        else:
            transforms.append(auto_augment_transform(auto_augment, aa_params))
    elif random_color_jitter:
        transforms.append(T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])
    return T.Compose(transforms)


def get_val_transform(resizing='default', resize_size=224,
                      norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
    """
    resizing mode:
        - default: resize the image to 256 and take the center crop of size 224;
        - res.: resize the image to 224
    """
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
    else:
        raise NotImplementedError(resizing)
    return T.Compose([
        transform,
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])

class MNIST():

    def __init__(self,path,opt = {}):
        self.path = path 
        self.opt = opt
        
    def get_loaders(self, train_transform_args = {}, val_transform_args = {}):

        opt = self.opt

        os.makedirs(self.path, exist_ok=True)

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                self.path,
                train=True,
                download=True,
                transform=T.Compose(
                    [T.Resize(opt["img_size"]), T.ToTensor(), T.Normalize([0.5], [0.5])]
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
                transform=T.Compose(
                    [T.Resize(opt["img_size"]),
                    T.ToTensor(),
                    T.Normalize([0.5], [0.5])
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
        
    def get_loaders(self, train_transform_args = {}, val_transform_args = {}):

        opt = self.opt

        os.makedirs(self.path, exist_ok=True)
        train_loader = torch.utils.data.DataLoader(
            MNISTM(
                "data/mnistm",
                train=True,
                download=True,
                transform=T.Compose(
                    [
                        T.Resize(opt["img_size"]),
                        T.ToTensor(),
                        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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
                transform=T.Compose(
                    [
                        T.Resize(opt["img_size"]),
                        T.ToTensor(),
                        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                ),
            ),
            batch_size=opt["batch_size"],
            shuffle=True,
        )

        return train_loader, test_loader

class VISDA17_real:

    def __init__(self,path,opt = {}): 
        self.path = path
        self.opt = opt 

    def get_loaders(self, train_transform_args = {}, val_transform_args = {}):

        opt = self.opt

        train_transform = get_train_transform(**train_transform_args)
        test_transform = get_val_transform(**val_transform_args)
        dataset = VisDA2017(self.path, 'Real', download=True)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train, test = torch.utils.data.random_split(dataset, [train_size, test_size])

        train.transform = train_transform
        test.transform = test_transform

        train_loader = DataLoader(train, batch_size=opt["batch_size"], shuffle=True, drop_last=True)
        test_loader = DataLoader(test, batch_size=opt["batch_size"], shuffle=False)

        return train_loader, test_loader

class VISDA17_synthetic:

    def __init__(self,path,opt = {}): 
        self.path = path
        self.opt = opt 

    def get_loaders(self, train_transform_args = {}, val_transform_args = {} ):

        opt = self.opt

        train_transform = get_train_transform(**train_transform_args)
        test_transform = get_val_transform(**val_transform_args)
        dataset = VisDA2017(self.path, 'Synthetic', download=True)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train, test = torch.utils.data.random_split(dataset, [train_size, test_size])

        train.transform = train_transform
        test.transform = test_transform

        train_loader = DataLoader(train, batch_size=opt["batch_size"], shuffle=True, drop_last=True)
        test_loader = DataLoader(test, batch_size=opt["batch_size"], shuffle=False)

        return train_loader, test_loader

