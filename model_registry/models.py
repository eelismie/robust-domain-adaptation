import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torchvision.transforms as transforms
import torchvision.models as mod
import timm
import tllib.vision.models as models_

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from tllib.self_training.mcc import ImageClassifier
from tllib.alignment.mdd import ImageClassifier as ImageClassifiermdd
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights

def get_model(model_name, pretrain=True):
    if model_name in models_.__dict__:
        # load models_ from tllib.vision.models_
        backbone = models_.__dict__[model_name](pretrained=pretrain)
    else:
        # load models from pytorch-image-models
        backbone = timm.create_model(model_name, pretrained=pretrain)
        try:
            backbone.out_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0, '')
        except:
            backbone.out_features = backbone.head.in_features
            backbone.head = nn.Identity()
    return backbone

class tlibClassifier(nn.Module):
    def __init__(self, opt=None) -> None:
        super(tlibClassifier, self).__init__()
        backbone = get_model(opt["arch"], pretrain=opt["pretrain"])
        pool_layer = nn.Identity() if opt["no_pool"] else None 
        self.model = ImageClassifier(backbone, opt["n_classes"], pool_layer=pool_layer,finetune=opt["pretrain"], bottleneck_dim=opt["bottleneck_dim"])

    def forward(self, x):
        out = self.model(x)
        return out

class tlibClassifiermdd(nn.Module):
    def __init__(self, opt=None) -> None:
        super(tlibClassifiermdd, self).__init__()
        backbone = get_model(opt["arch"], pretrain=opt["pretrain"])
        pool_layer = nn.Identity() if opt["no_pool"] else None
        self.model = ImageClassifiermdd(backbone, opt["n_classes"], pool_layer=pool_layer, finetune=opt["pretrain"], bottleneck_dim=opt["bottleneck_dim"], width=1024)

    def forward(self, x):
        out = self.model(x)
        return out