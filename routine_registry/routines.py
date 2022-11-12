import numpy as np
import os
import itertools
import plotly.express as px 

from torch.autograd import Variable
from torchattacks import PGD
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from tllib.utils.data import ForeverDataIterator
from tllib.alignment.mdd import ClassificationMarginDisparityDiscrepancy as MarginDisparityDiscrepancy
from tllib.self_training.mcc import MinimumClassConfusionLoss
from torchmetrics import ConfusionMatrix, Accuracy
from tqdm import tqdm

import torch.nn.functional as F
import torch

import wandb

cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plotly_confusion_matrix(confusion_matrix, class_names):
    fig = px.imshow(np.around(confusion_matrix,3), x=class_names, y=class_names, text_auto=True)
    return fig

def validate(val_loader, model, args):

    model.eval()
    confmat = ConfusionMatrix(len(args["class_names"]), normalize='true').to(device)
    accuracy = Accuracy(len(args["class_names"])).to(device)

    max_iters = int(args["n_classes"]*500/args["batch_size"])

    with torch.no_grad():
        for i, data in tqdm(enumerate(val_loader)):
            images, labels = data 
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)                 
            output = torch.argmax(outputs, 1) 
            accuracy.update(output, labels)
            confmat.update(output, labels)
            if (i >= max_iters):
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

    confmat = ConfusionMatrix(len(args["class_names"]), normalize='true').to(device)
    accuracy = Accuracy(len(args["class_names"])).to(device)

    max_iters = int(args["n_classes"]*500/args["batch_size"])

    atk = PGD(model, eps=8/255, alpha=2/225, steps=10, random_start=True)
    #assume standard normalization TODO: infer this from val loader
    atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    for i, data in tqdm(enumerate(val_loader)):
        images, labels = data 
        images = images.to(device)
        labels = labels.to(device)
        adv_images = atk(images, labels)
        outputs = model(adv_images)                 
        output = torch.argmax(outputs, 1) 
        accuracy.update(output, labels)
        confmat.update(output, labels)
        if (i >= max_iters):
            break

    acc = accuracy.compute()
    confusion = confmat.compute()

    return acc.cpu().numpy(), confusion.cpu().numpy()

def train_mcc(
    classifier = None,
    source_train = None,
    target_train = None,
    source_val = None,
    target_val = None,
    opt = None
    ):

    """
    modified from mcc.py in ttlib / transfer learning library

    Original author: 
    @author: Junguang Jiang
    @contact: JiangJunguang1123@outlook.com
    """

    wandb.init(project="robust-domain-adaptation", config=opt)

    classifier.train()

    train_source_iter = ForeverDataIterator(source_train)
    train_target_iter = ForeverDataIterator(target_train)

    classifier = classifier.to(device)

    # define optimizer and lr scheduler
    optimizer = SGD(classifier.model.get_parameters(), opt["lr"], momentum=opt["momentum"], weight_decay=opt["weight_decay"],nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: opt["lr"] * (1. + opt["lr_gamma"] * float(x)) ** (-opt["lr_decay"]))
    mcc_loss = MinimumClassConfusionLoss(temperature=opt["temperature"])

    for e in range(opt["num_epochs"]):
        for i in range(opt["iters_per_epoch"]):
            x_s, labels_s = next(train_source_iter)[:2]
            x_t, = next(train_target_iter)[:1]

            x_s = x_s.to(device)
            x_t = x_t.to(device)
            labels_s = labels_s.to(device)

            # compute output
            x = torch.cat((x_s, x_t), dim=0)
            y, f = classifier(x)
            y_s, y_t = y.chunk(2, dim=0)

            cls_loss = F.cross_entropy(y_s, labels_s)
            transfer_loss = mcc_loss(y_t)
            loss = cls_loss + transfer_loss * opt["trade_off"]

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        if (e % 2 == 0):

            global_acc_a, confusion_a = validate(source_val, classifier, opt)
            global_acc_b, confusion_b = validate(target_val, classifier, opt)

            per_class_a = np.diag(confusion_a).tolist()
            per_class_b = np.diag(confusion_b).tolist()

            logged_metrics = [ ("validation accuracy (target)", global_acc_b), ("validation accuracy (source)", global_acc_a) ]
            logged_metrics += [ (pair[0] + "_target", pair[1]) for pair in zip(opt["class_names"], per_class_b) ]
            logged_metrics += [ (pair[0] + "_source", pair[1]) for pair in zip(opt["class_names"], per_class_a) ]

            log_ = { pair[0] : pair[1] for pair in logged_metrics }

            wandb.log(log_)

    global_acc_a, confusion_a = validate(source_val, classifier, opt)
    global_acc_b, confusion_b = validate(target_val, classifier, opt)

    f_a = plotly_confusion_matrix(confusion_a, opt["class_names"])
    f_b = plotly_confusion_matrix(confusion_b, opt["class_names"])

    wandb.log({"source confusion": wandb.Plotly(f_a)})
    wandb.log({"target confusion": wandb.Plotly(f_b)})

    torch.save({
        'epoch': opt["num_epochs"],
        'model_state_dict': classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(opt["checkpoint_path"], "mdd.pt"))


def train_mdd(
    classifier = None,
    source_train = None,
    target_train = None,
    source_val = None,
    target_val = None,
    opt = None
    ):

    """
    modified from mdd.py in ttlib / transfer learning library
    
    Original author: 
    @author: Junguang Jiang
    @contact: JiangJunguang1123@outlook.com
    """

    wandb.init(project="robust-domain-adaptation", config=opt)

    classifier.train()

    train_source_iter = ForeverDataIterator(source_train)
    train_target_iter = ForeverDataIterator(target_train)

    classifier = classifier.to(device)

    # define optimizer and lr scheduler
    mdd = MarginDisparityDiscrepancy(opt["margin"]).to(device)
    optimizer = SGD(classifier.model.get_parameters(), opt["lr"], momentum=opt["momentum"], weight_decay=opt["weight_decay"], nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: opt["lr"] * (1. + opt["lr_gamma"] * float(x)) ** (-opt["lr_decay"]))

    for e in range(opt["num_epochs"]):
        for i in range(opt["iters_per_epoch"]):

            optimizer.zero_grad()

            x_s, labels_s = next(train_source_iter)[:2]
            x_t, = next(train_target_iter)[:1]

            x_s = x_s.to(device)
            x_t = x_t.to(device)
            labels_s = labels_s.to(device)

            # compute output
            x = torch.cat((x_s, x_t), dim=0)
            outputs, outputs_adv = classifier(x)
            y_s, y_t = outputs.chunk(2, dim=0)
            y_s_adv, y_t_adv = outputs_adv.chunk(2, dim=0)

            # compute cross entropy loss on source domain
            cls_loss = F.cross_entropy(y_s, labels_s)
            # compute margin disparity discrepancy between domains
            # for adversarial classifier, minimize negative mdd is equal to maximize mdd
            transfer_loss = -mdd(y_s, y_s_adv, y_t, y_t_adv)
            loss = cls_loss + transfer_loss * opt["trade_off"]
            classifier.model.step()

            # compute gradient and do SGD step
            loss.backward() 
            optimizer.step()
            lr_scheduler.step()

        if (e % 2 == 0):

            global_acc_a, confusion_a = validate(source_val, classifier, opt)
            global_acc_b, confusion_b = validate(target_val, classifier, opt)

            per_class_a = np.diag(confusion_a).tolist()
            per_class_b = np.diag(confusion_b).tolist()

            logged_metrics = [ ("validation accuracy (target)", global_acc_b), ("validation accuracy (source)", global_acc_a) ]
            logged_metrics += [ (pair[0] + "_target", pair[1]) for pair in zip(opt["class_names"], per_class_b) ]
            logged_metrics += [ (pair[0] + "_source", pair[1]) for pair in zip(opt["class_names"], per_class_a) ]

            log_ = { pair[0] : pair[1] for pair in logged_metrics }

            wandb.log(log_)

    global_acc_a, confusion_a = validate(source_val, classifier, opt)
    global_acc_b, confusion_b = validate(target_val, classifier, opt)

    f_a = plotly_confusion_matrix(confusion_a, opt["class_names"])
    f_b = plotly_confusion_matrix(confusion_b, opt["class_names"])

    wandb.log({"source confusion": wandb.Plotly(f_a)})
    wandb.log({"target confusion": wandb.Plotly(f_b)})

    torch.save({
        'epoch': opt["num_epochs"],
        'model_state_dict': classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(opt["checkpoint_path"], "mdd.pt"))


def check_adv_accuracy(
    classifier = None,
    test_loader = None, 
    opt = None
    ):

    classifier.to(device)
    #TODO: infer the normalization used from dataset
    atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    atk = PGD(classifier, eps=8/255, alpha=2/225, steps=10, random_start=True)

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        adv_images = atk(images, labels)

    return 
