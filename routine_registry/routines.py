import numpy as np
import os
import plotly.express as px 

from torchattacks import PGD
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from tllib.utils.data import ForeverDataIterator
from tllib.alignment.mdd import ClassificationMarginDisparityDiscrepancy as MarginDisparityDiscrepancy
from tllib.self_training.mcc import MinimumClassConfusionLoss
from torchmetrics import ConfusionMatrix, Accuracy
from tqdm import tqdm
from .nwd import NuclearWassersteinDiscrepancy
from datetime import datetime

import torch.nn.functional as F
import torch

import wandb

cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

def set_global_seed(args):
    torch.backends.cudnn.deterministic = True
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])

def plotly_confusion_matrix(confusion_matrix, class_names):
    fig = px.imshow(np.around(confusion_matrix,3), x=class_names, y=class_names, text_auto=True)
    return fig

def validate(val_loader, model, args):

    model.eval()
    confmat = ConfusionMatrix(len(args["class_names"]), normalize='true').to(device)
    accuracy = Accuracy(len(args["class_names"])).to(device)

    with torch.no_grad():
        for i, data in tqdm(enumerate(val_loader)):
            images, labels = data 
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)                 
            output = torch.argmax(outputs, 1) 
            accuracy.update(output, labels)
            confmat.update(output, labels)
            if (i >= args["iters_per_epoch"]):
                break

    model.train()
    acc = accuracy.compute()
    confusion = confmat.compute()
    return acc.cpu().numpy(), confusion.cpu().numpy()

def validate_adv(
    val_loader = None, 
    model = None,
    args = None):

    """
    For the validation set : compute adversarial inputs and measure accuracy
    """

    model.eval()

    confmat = ConfusionMatrix(len(args["class_names"]), normalize='true').to(device)
    accuracy = Accuracy(len(args["class_names"])).to(device)

    atk = PGD(model, eps=1/500, alpha=1/500, steps=6, random_start=True)
    #assume standard normalization TODO: infer this from val loader
    atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    for i, data in  enumerate(val_loader):
        images, labels = data 
        images = images.to(device)
        labels = labels.to(device)
        adv_images = atk(images, labels)
        pred = model(adv_images)
        output = torch.argmax(pred, 1) 
        accuracy.update(output, labels)
        confmat.update(output, labels)
        if (i >= args["adv_validation_iters"]):
            break

    acc = accuracy.compute()
    confusion = confmat.compute()

    return acc.cpu().numpy(), confusion.cpu().numpy()

class experiment:

    def __init__(
        self,
        classifier = None,
        source_train = None,
        target_train = None,
        source_val = None,
        target_val = None,
        opt = None
        ):

        self.classifier = classifier
        self.source_train = source_train
        self.target_train = target_train
        self.source_val = source_val
        if (target_val):
            self.target_val = target_val
        else:
            self.target_val = target_train
        self.opt = opt

    def run(self):
        self.setup()
        for e in range(self.opt["num_epochs"]):
            self.each_epoch(e)
        self.cleanup()

    def each_epoch(self, e):
        # define actions before and after each epoch
        for i in range(self.opt["iters_per_epoch"]):
            self.each_iter(i) 

    def setup(self):
        set_global_seed(self.opt)
        wandb.init(project="robust-domain-adaptation", config=self.opt)
        self.classifier.train()
        self.classifier = self.classifier.to(device)
        self.optimizer = SGD(self.classifier.model.get_parameters(), self.opt["lr"], momentum=self.opt["momentum"], weight_decay=self.opt["weight_decay"],nesterov=True)
        self.lr_scheduler = LambdaLR(self.optimizer, lambda x: self.opt["lr"] * (1. + self.opt["lr_gamma"] * float(x)) ** (-self.opt["lr_decay"]))

    def cleanup(self):

        global_acc_a, confusion_a = validate(self.source_val, self.classifier, self.opt)
        global_acc_b, confusion_b = validate(self.target_val, self.classifier, self.opt)

        f_a = plotly_confusion_matrix(confusion_a, self.opt["class_names"])
        f_b = plotly_confusion_matrix(confusion_b, self.opt["class_names"])

        wandb.log({"source confusion": wandb.Plotly(f_a)})
        wandb.log({"target confusion": wandb.Plotly(f_b)})
        wandb.log({"source validation accuracy": global_acc_a})
        wandb.log({"target validation accuracy": global_acc_b})

        now = datetime.now()
        current_time = now.strftime("%D:%H:%M:%S").replace("/","").replace(":","")
        model_path = current_time + "_mdd.pt"

        return

    def each_iter(self, i):
        # define actions during each iteratos
        return

class mcc_experiment(experiment):

    """
    modified from mcc.py in ttlib / transfer learning library
    
    Original author: 
    @author: Junguang Jiang
    @contact: JiangJunguang1123@outlook.com
    """

    def setup(self):
        super().setup()
        self.train_source_iter = ForeverDataIterator(self.source_train)
        self.train_target_iter = ForeverDataIterator(self.target_train)
        self.mcc_loss = MinimumClassConfusionLoss(temperature=self.opt["temperature"])

    def each_iter(self, i):
        x_s, labels_s = next(self.train_source_iter)[:2]
        x_t, = next(self.train_target_iter)[:1]

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)

        # compute output
        x = torch.cat((x_s, x_t), dim=0)
        y, f = self.classifier(x)
        y_s, y_t = y.chunk(2, dim=0)

        cls_loss = F.cross_entropy(y_s, labels_s)
        transfer_loss = self.mcc_loss(y_t)
        loss = cls_loss + transfer_loss * self.opt["trade_off"]

        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

    def each_epoch(self, e):
        super().each_epoch(e)

        global_acc_a, confusion_a = validate(self.source_val, self.classifier, self.opt)
        global_acc_b, confusion_b = validate(self.target_val, self.classifier, self.opt)

        per_class_a = np.diag(confusion_a).tolist()
        per_class_b = np.diag(confusion_b).tolist()

        logged_metrics = [ ("validation accuracy (target)", global_acc_b), ("validation accuracy (source)", global_acc_a) ]
        logged_metrics += [ (pair[0] + "_target", pair[1]) for pair in zip(self.opt["class_names"], per_class_b) ]
        logged_metrics += [ (pair[0] + "_source", pair[1]) for pair in zip(self.opt["class_names"], per_class_a) ]

        log_ = { pair[0] : pair[1] for pair in logged_metrics }

        wandb.log(log_)

    def cleanup(self):
        super().cleanup()
        return

class mdd_experiment(experiment):

    """
    modified from mdd.py in ttlib / transfer learning library
    
    Original author: 
    @author: Junguang Jiang
    @contact: JiangJunguang1123@outlook.com
    """

    def setup(self):
        super().setup()
        self.train_source_iter = ForeverDataIterator(self.source_train)
        self.train_target_iter = ForeverDataIterator(self.target_train)
        self.mdd = MarginDisparityDiscrepancy(self.opt["margin"]).to(device)

    def each_iter(self, i):
        self.optimizer.zero_grad()

        x_s, labels_s = next(self.train_source_iter)[:2]
        x_t, = next(self.train_target_iter)[:1]

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)

        # compute output
        x = torch.cat((x_s, x_t), dim=0)
        outputs, outputs_adv = self.classifier(x)
        y_s, y_t = outputs.chunk(2, dim=0)
        y_s_adv, y_t_adv = outputs_adv.chunk(2, dim=0)

        # compute cross entropy loss on source domain
        cls_loss = F.cross_entropy(y_s, labels_s)
        # compute margin disparity discrepancy between domains
        # for adversarial classifier, minimize negative mdd is equal to maximize mdd
        transfer_loss = -self.mdd(y_s, y_s_adv, y_t, y_t_adv)
        loss = cls_loss + transfer_loss * self.opt["trade_off"]

        # compute gradient and do SGD step
        loss.backward() 
        self.optimizer.step()
        self.lr_scheduler.step()

    def cleanup(self):
        super().cleanup()
        return

class daln_experiment(experiment):

    def setup(self):
        super().setup()
        self.train_source_iter = ForeverDataIterator(self.source_train)
        self.train_target_iter = ForeverDataIterator(self.target_train)
        self.discrepancy = NuclearWassersteinDiscrepancy(self.classifier.model.head).to(device)

    def each_iter(self, i):
        self.optimizer.zero_grad()

        x_s, labels_s = next(self.train_source_iter)[:2]
        x_t, = next(self.train_target_iter)[:1]

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)

        # compute output
        x = torch.cat((x_s, x_t), dim=0)
        y, f = self.classifier(x)
        y_s, y_t = y.chunk(2, dim=0)

        cls_loss = F.cross_entropy(y_s, labels_s)
        discrepancy_loss = -self.discrepancy(f)
        transfer_loss = discrepancy_loss * self.opt["trade_off"]
        loss = cls_loss + transfer_loss

        # compute gradient and do SGD step
        loss.backward() 
        self.optimizer.step()
        self.lr_scheduler.step()


def train_mcc(
    classifier = None,
    source_train = None,
    target_train = None,
    source_val = None,
    target_val = None,
    opt = None
    ):

    experiment_ = mcc_experiment(
        classifier=classifier, 
        source_train=source_train, 
        target_train=target_train,
        source_val=source_val,
        target_val=None
        opt=opt) 

    experiment_.run()
    return



def train_mdd(
    classifier = None,
    source_train = None,
    target_train = None,
    source_val = None,
    target_val = None,
    opt = None
    ):

    experiment_ = mdd_experiment(
        classifier=classifier,
        source_train=source_train,
        target_train=target_train,
        source_val=source_val,
        target_val=None,
        opt=opt
    )

    experiment_.run()
    return

def train_daln(
    classifier = None,
    source_train = None,
    target_train = None,
    source_val = None,
    target_val = None,
    opt = None
    ):

    experiment_ = daln_experiment(
        classifier=classifier,
        source_train=source_train,
        target_train=target_train,
        source_val=source_val,
        target_val=None,
        opt=opt
    )

    experiment_.run()
    return  