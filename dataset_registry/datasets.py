import torch
import torchvision.transforms as T
import torch.utils.data
import numpy as np
import torch.nn.functional as F

from typing import Optional
from torchvision import datasets
from torch.utils.data import DataLoader
from tllib.vision.transforms import ResizeImage
from tllib.vision.datasets.visda2017 import VisDA2017
from tllib.vision.datasets.digits import SVHN, USPS
from tllib.vision.datasets.office31 import Office31
from tllib.vision.datasets.pacs import PACS
from timm.data.auto_augment import auto_augment_transform, rand_augment_transform
from torch.utils.data.sampler import Sampler

def get_train_transform(resizing='default', scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), random_horizontal_flip=True,
                        random_color_jitter=False, resize_size=224, norm_mean=(0.485, 0.456, 0.406),
                        norm_std=(0.229, 0.224, 0.225), auto_augment=None, no_change=False):
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
    if (no_change):
        return T.Compose([T.ToTensor()])
    return T.Compose(transforms)


def get_val_transform(resizing='default', resize_size=224,
                      norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225), no_change=False):
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

    if (no_change):
        return T.Compose([T.ToTensor()])
    return T.Compose([transform,
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])

class visdaAdapter(VisDA2017):
    def __init__(self, root: str, task: str, split='all', download: Optional[bool] = True, **kwargs):
        super().__init__(root, task, split=split, download = download, **kwargs)
        self.targets = np.array(self.targets) #convert targets to numpy array
        self.class_indices = [(self.targets == class_id).nonzero()[0] for class_id in range(self.num_classes)]
        self.sampler = ClassSampler(self, gamma=0.5)

class pacsAdapter(PACS):
    def __init__(self, root: str, task: str, split='all', download: Optional[bool] = True, **kwargs):
        super().__init__(root, task, split=split, download = download, **kwargs)
        self.targets = np.array(self.targets) #convert targets to numpy array
        self.class_indices = [(self.targets == class_id).nonzero()[0] for class_id in range(self.num_classes)]
        self.sampler = ClassSampler(self, gamma=0.5)

class ClassSampler(Sampler[int]):

    """
    Args:
        dataset (Dataset): dataset to sample from
    """

    def __init__(self, 
        dataset, 
        gamma=1/2, 
        base_dist="uniform", 
        prior="uniform",
        reweight=False,
    ) -> None:

        self.gamma = gamma
        self.dataset = dataset
        self.reweight = reweight
        
        uniform = 1/(self.dataset.num_classes) * torch.ones(dataset.num_classes)
        self._class_fractions = torch.tensor([len(indices) for indices in dataset.class_indices]) / len(dataset)

        if base_dist == 'uniform':
            self.base_dist = uniform
        elif base_dist == 'empirical':
            self.base_dist = self._class_fractions
        else:
            raise ValueError("Invalid ClassSampler.base_dist")

        # Initialize prior
        self.prior = prior
        self.reset_prior()

    def reset_prior(self):
        if self.prior == 'uniform':
            self.w = torch.zeros(self.dataset.num_classes)
        elif self.prior == 'empirical':
            self.w = torch.log(self._class_fractions)
        elif self.prior is None:        
            self.w = torch.log(self.base_dist)
        else:
            raise ValueError("Invalid ClassSampler.prior")

    @property
    def q(self):
        return F.softmax(self.w, dim=0)

    @property
    def p(self):
        # Only sample nonuniformly if self.reweight=False
        return self.distribution(use_base=self.reweight)

    def batch_weight(self, class_ids):
        # Only weight nonuniformly if self.reweight=True
        p = self.distribution(use_base=not self.reweight)
        return p[class_ids]

    def distribution(self, use_base):
        if use_base:
            return self.base_dist
        else:
            return self.gamma * self.base_dist + (1-self.gamma) * self.q

    def batch_update(self, class_ids, eta_times_loss_arms):
        """Parallel update (does not update self.p sequentially)
        """
        loss_vec = torch.zeros(self.dataset.num_classes)
        p = self.p
        for i, class_id in enumerate(class_ids):
            eta_times_loss_arm = eta_times_loss_arms[i]
            loss_vec[class_id] = loss_vec[class_id] + eta_times_loss_arm / p[class_id]
        self.w = self.w + loss_vec

    def update(self, class_id, eta_times_loss_arm):
        loss_vec = torch.zeros(self.dataset.num_classes)
        loss_vec[class_id] = eta_times_loss_arm / self.p[class_id]
        self.w = self.w + loss_vec

    def sample_class_id(self):
        class_id = torch.multinomial(self.p, num_samples=1).item()
        return class_id

    def __iter__(self):
        for _ in range(len(self)):
            class_id = self.sample_class_id()
            class_indices = self.dataset.class_indices[class_id]
            idx = torch.randint(high=len(class_indices), size=(1,), 
                                dtype=torch.int64).item()
            yield class_indices[idx]

    def __len__(self):
        return len(self.dataset)

class domainDataset:

    def __init__(self, path, opt = {}) -> None:
        raise(NotImplementedError)
        
    def get_loaders(self, train_transform_args = {}, val_transform_args = {}):

        opt = self.opt

        train_transform = get_train_transform(**train_transform_args)
        test_transform = get_val_transform(**val_transform_args)
        dataset = VisDA2017(self.path, 'Real', download=True)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train, test = torch.utils.data.random_split(dataset, [train_size, test_size])

        train.dataset.transform = train_transform
        test.dataset.transform = test_transform

        train_loader = DataLoader(train, batch_size=opt["batch_size"], shuffle=True, drop_last=True)
        test_loader = DataLoader(test, batch_size=opt["batch_size"], shuffle=False)

        return train_loader, test_loader

class VISDA17_real(domainDataset):

    def __init__(self,path,opt = {}): 
        self.path = path
        self.opt = opt 
        self.dataset = VisDA2017(self.path, 'Real', download=True)
        self.class_names = self.dataset.classes
        self.n_classes = len(self.class_names)

    #create new classes on top of VisDA2017 etc. that store the indices of each class 

class VISDA17_synthetic(domainDataset):

    def __init__(self,path,opt = {}): 
        self.path = path
        self.opt = opt 
        self.dataset = VisDA2017(self.path, 'Synthetic', download=True)
        self.class_names = self.dataset.classes
        self.n_classes = len(self.class_names)


class PACS_P(domainDataset):

    def __init__(self,path,opt = {}): 
        self.path = path
        self.opt = opt 
        self.dataset = PACS(self.path, task="P", split="all", download=True)
        self.class_names = self.dataset.classes
        self.n_classes = len(self.class_names)


class PACS_A(domainDataset):

    def __init__(self,path,opt = {}): 
        self.path = path
        self.opt = opt 
        self.dataset = PACS(self.path, task="A", split="all", download=True)
        self.class_names = self.dataset.classes
        self.n_classes = len(self.class_names)
    #create new classes on top of VisDA2017 etc. that store the indices of each class 

class PACS_C(domainDataset):

    def __init__(self,path,opt = {}): 
        self.path = path
        self.opt = opt 
        self.dataset = PACS(self.path, task="C", split="all", download=True)
        self.class_names = self.dataset.classes
        self.n_classes = len(self.class_names)


