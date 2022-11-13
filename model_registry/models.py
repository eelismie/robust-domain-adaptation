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
        self.model = ImageClassifier(backbone, opt["n_classes"], finetune=opt["pretrain"], bottleneck_dim=1024)

    def forward(self, x):
        out = self.model(x)
        return out

class tlibClassifiermdd(nn.Module):
    def __init__(self, opt=None) -> None:
        super(tlibClassifiermdd, self).__init__()
        backbone = get_model(opt["arch"], pretrain=opt["pretrain"])
        self.model = ImageClassifiermdd(backbone, opt["n_classes"], finetune=opt["pretrain"], bottleneck_dim=1024, width=1024)

    def forward(self, x):
        out = self.model(x)
        return out

class simpleClassifier(nn.Module):
    def __init__(self, opt=None):
        super(simpleClassifier, self).__init__()

        def block(in_features, out_features, normalization=True):
            """Classifier block"""
            layers = [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_features))
            return layers

        self.model = nn.Sequential(
            *block(opt["channels"], 64, normalization=False), *block(64, 128), *block(128, 256), *block(256, 512)
        )

        input_size = opt["img_size"] // 2 ** 4
        self.output_layer = nn.Sequential(nn.Linear(512 * input_size ** 2, opt["n_classes"]), nn.Softmax())

    def forward(self, img):
        feature_repr = self.model(img)
        feature_repr = feature_repr.view(feature_repr.size(0), -1)
        label = self.output_layer(feature_repr)
        return label

class resnet18classifier(nn.Module):
    def __init__(self, opt={}):
        super(resnet18classifier, self).__init__()
        weights = None 
        if opt["pretrain"]:
            weights = ResNet18_Weights.DEFAULT
        self.backbone = mod.resnet18(weights=weights)
        self.head = nn.Linear(self.backbone.fc.out_features, opt["n_classes"])

    def forward(self, x):
        out = self.head(self.backbone(x))
        return out

class resnet50classifier(nn.Module):
    def __init__(self, opt={}):
        super(resnet50classifier, self).__init__()
        weights = None 
        if opt["pretrain"]:
            weights = ResNet50_Weights.DEFAULT
        self.backbone = mod.resnet50(weights=weights)
        self.head = nn.Linear(self.backbone.fc.out_features, opt["n_classes"])

    def forward(self, x):
        out = self.head(self.backbone(x))
        return out

class resnet101classifier(nn.Module):
    def __init__(self, opt={}):
        super(resnet101classifier, self).__init__()
        self.backbone = mod.resnet101(pretrained=opt["pretrain"])
        self.head = nn.Linear(self.backbone.fc.out_features, opt["n_classes"])

    def forward(self, x):
        out = self.head(self.backbone(x))
        return out
