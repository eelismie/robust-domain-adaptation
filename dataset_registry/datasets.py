import torch
import torchvision.transforms as T
import torch.utils.data
import numpy as np
import copy
import torch.nn.functional as F

from PIL import Image
from typing import Tuple
from torch.utils.data import DataLoader
from tllib.vision.transforms import ResizeImage
from tllib.vision.datasets.visda2017 import VisDA2017
from tllib.vision.datasets.pacs import PACS
from tllib.vision.datasets.domainnet import DomainNet
from timm.data.auto_augment import auto_augment_transform, rand_augment_transform
from torch.utils.data.sampler import Sampler
from tllib.vision.datasets import ImageList
from sklearn.model_selection import train_test_split

"""
This package contains utility classes for loading VISDA, PACS, and DomainNet datasets. 
"""

def get_train_transform(resizing='default', scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), random_horizontal_flip=True,
                        random_color_jitter=False, resize_size=224, norm_mean=(0.485, 0.456, 0.406),
                        norm_std=(0.229, 0.224, 0.225), auto_augment=None, no_change=False):
    """
    resizing mode:
        - default: resize the image to 256 and take a random resized crop of size 224;
        - cen.crop: resize the image to 256 and take the center crop of size 224;
        - res: resize the image to 224;

    Modified from ttlib library:
    @author: Junguang Jiang, Baixu Chen
    @contact: JiangJunguang1123@outlook.com, cbx_99_hasta@outlook.com
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

    Modified from ttlib library:
    @author: Junguang Jiang, Baixu Chen
    @contact: JiangJunguang1123@outlook.com, cbx_99_hasta@outlook.com
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
        
def stratifiedSplit(dataset: ImageList, test_size=0.25, random_state=42) -> Tuple[ImageList, ImageList]:
        
        """
        Create stratified train / test split for ttlib ImageList Objects. 
        Train and Test datasets are first created as copies of the base dataset.
        Then, since the datasets are simply lists of image paths and their targets, 
        the split can be performed by splitting the list of samples.  
        """
        
        samples = dataset.samples #list tuple(path to image, name)
        samples_train, samples_test = train_test_split(
                samples,
                test_size=test_size,
                random_state=random_state,
                stratify=np.array([t[1] for t in samples])
                )

        train_set = copy.deepcopy(dataset)
        test_set = copy.deepcopy(dataset)

        train_set.samples = samples_train
        train_set.targets = [s[1] for s in train_set.samples]
        test_set.samples = samples_test
        test_set.targets = [s[1] for s in test_set.samples]

        assert(len(set(train_set.samples).intersection(set(test_set.samples))) == 0) 

        return train_set, test_set

class ClassSampler(Sampler[int]):

    """
    Args:
        dataset (Dataset): dataset to sample from

    Modified from: 
    @author : Thomas Pethick 
    @contact : thomas.pethick@epfl.ch 
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

    """
    Base template for all datasets 
    Extensions of this class define how the train and test sets are built and split. 
    """

    def __init__(self, path, opt = {}) -> None:
        self.opt = None
        self.path = None
        self.train_dataset = None
        self.test_dataset = None
        raise(NotImplementedError)
        
    def get_loaders(self, train_transform_args = {}, val_transform_args = {}):

        opt = self.opt

        train_transform = get_train_transform(**train_transform_args)
        test_transform = get_val_transform(**val_transform_args)

        self.train_dataset.transform = train_transform
        self.test_dataset.transform = test_transform

        train_loader = None
        if (opt["cfol_sampling"]):
            self.train_dataset = self.insertClassIndices(self.train_dataset)
            sampler = ClassSampler(self.train_dataset, gamma=0.5)
            train_loader = DataLoader(self.train_dataset, batch_size=opt["batch_size"], drop_last=True, sampler=sampler)
        else:
            train_loader = DataLoader(self.train_dataset, batch_size=opt["batch_size"], shuffle=True, drop_last=True)

        test_loader = DataLoader(self.test_dataset, batch_size=opt["batch_size"], drop_last=True)

        return train_loader, test_loader

    def insertClassIndices(dataset: ImageList) -> ImageList:
        # required for a dataset to be compatible with cfol sampler
        if not isinstance(dataset.targets, np.ndarray):
            dataset.targets = np.array(dataset.targets)
        dataset.class_indices = [(dataset.targets == class_id).nonzero()[0] for class_id in range(dataset.num_classes)]
        return dataset

class VISDA17_real(domainDataset):

    """
    Real objects in VISDA
    """

    def __init__(self,path,opt = {}): 
        self.path = path
        self.opt = opt 
        self.dataset = VisDA2017(self.path, 'Real', download=True)

        train, test = stratifiedSplit(self.dataset, self.opt["test_size"], self.opt["seed"])
        self.train_dataset = train
        self.test_dataset = test
        self.class_names = self.dataset.classes
        self.n_classes = len(self.class_names)

class VISDA17_synthetic(domainDataset):

    """
    Synthetic objects in VISDA
    """

    def __init__(self,path,opt = {}): 
        self.path = path
        self.opt = opt 
        self.dataset = VisDA2017(self.path, 'Synthetic', download=True)

        train, test = stratifiedSplit(self.dataset, self.opt["test_size"], self.opt["seed"])
        self.train_dataset = train
        self.test_dataset = test
        self.class_names = self.dataset.classes
        self.n_classes = len(self.class_names)

class DNET_R(domainDataset):

    """
    Domainnet real photos
    """

    def __init__(self, path, opt={}) -> None:
        self.path = path
        self.opt = opt 

        #use split that is already provided
        self.train_dataset = DomainNet(self.path, "r", split="train", download=True)
        self.test_dataset = DomainNet(self.path, "r", split="test", download=True)
        self.class_names = self.train_dataset.classes
        self.n_classes = len(self.class_names)

class DNET_S(domainDataset):

    """
    Domainnet sketches
    """

    def __init__(self, path, opt={}) -> None:
        self.path = path
        self.opt = opt 

        #use split that is already provided
        self.train_dataset = DomainNet(self.path, "s", split="train", download=True)
        self.test_dataset = DomainNet(self.path, "s", split="test", download=True)
        self.class_names = self.train_dataset.classes
        self.n_classes = len(self.class_names)

class PACS_A(domainDataset):

    "PACS Artistic renditions"

    def __init__(self, path, opt={}) -> None:
        self.path = path
        self.opt = opt 
        self.dataset = PACS(self.path, "A", split="all", download=True)

        train, test = stratifiedSplit(self.dataset, self.opt["test_size"], self.opt["seed"])
        self.train_dataset = train
        self.test_dataset = test
        self.class_names = self.dataset.classes
        self.n_classes = len(self.class_names)
        
class PACS_P(domainDataset):

    "PACS Photos"

    def __init__(self, path, opt={}) -> None:
        self.path = path
        self.opt = opt 
        self.dataset = PACS(self.path, "P", split="all", download=True)

        train, test = stratifiedSplit(self.dataset, self.opt["test_size"], self.opt["seed"])
        self.train_dataset = train
        self.test_dataset = test
        self.class_names = self.dataset.classes
        self.n_classes = len(self.class_names)