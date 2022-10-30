import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torchvision.transforms as transforms
import torchvision.models as mod
import timm

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights

class ResidualBlock(nn.Module):
    def __init__(self, in_features=64, out_features=64):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, 1, 1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3, 1, 1),
            nn.BatchNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class simpleGenerator(nn.Module):
    def __init__(self, opt=None):
        super(simpleGenerator, self).__init__()

        # Fully-connected layer which constructs image channel shaped output from noise
        self.fc = nn.Linear(opt["latent_dim"], opt["channels"] * opt["img_size"] ** 2)
        self.l1 = nn.Sequential(nn.Conv2d(opt["channels"] * 2, 64, 3, 1, 1), nn.ReLU(inplace=True))

        resblocks = []
        for _ in range(opt["n_residual_blocks"]):
            resblocks.append(ResidualBlock())
        self.resblocks = nn.Sequential(*resblocks)

        self.l2 = nn.Sequential(nn.Conv2d(64, opt["channels"], 3, 1, 1), nn.Tanh())

    def forward(self, img, z):
        gen_input = torch.cat((img, self.fc(z).view(*img.shape)), 1)
        out = self.l1(gen_input)
        out = self.resblocks(out)
        img_ = self.l2(out)

        return img_


class simpleDiscriminator(nn.Module):
    def __init__(self, opt=None):
        super(simpleDiscriminator, self).__init__()

        def block(in_features, out_features, normalization=True):
            """Discriminator block"""
            layers = [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_features))
            return layers

        self.model = nn.Sequential(
            *block(opt["channels"], 64, normalization=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.Conv2d(512, 1, 3, 1, 1)
        )

    def forward(self, img):
        validity = self.model(img)

        return validity


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
