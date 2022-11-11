import numpy as np
import itertools
import plotly.express as px 

from torch.autograd import Variable
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from tllib.utils.data import ForeverDataIterator
from tllib.alignment.mdd import ClassificationMarginDisparityDiscrepancy as MarginDisparityDiscrepancy

from tllib.self_training._loss import MinimumClassConfusionLoss
from torchmetrics import ConfusionMatrix, Accuracy

import torch.nn.functional as F
import torch

import wandb

cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plotly_confusion_matrix(confusion_matrix, class_names):
    fig = px.imshow(np.around(confusion_matrix,3), x=class_names, y=class_names, text_auto=True)
    return fig

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def validate(val_loader, model, args, device):

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    # switch to evaluate mode
    model.eval()
    confmat = ConfusionMatrix(len(args["class_names"]), normalize='true').to(device)
    accuracy = Accuracy(len(args["class_names"])).to(device)

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, target = data[:2]
            batch_size = images.size(0)
            images = Variable(images.type(FloatTensor).expand(batch_size, 3, args["img_size"], args["img_size"]))
            target = Variable(target.type(LongTensor))
            
            # compute output
            output = torch.argmax(model(images), 1) 

            # measure accuracy and record loss
            accuracy.update(output, target)
            confmat.update(output, target)

    model.train()
    acc = accuracy.compute()
    confusion = confmat.compute()
    return acc.cpu().numpy(), confusion.cpu().numpy()

def train_pixel_da(
        generator = None, 
        discriminator = None, 
        classifier = None, 
        source_train = None, 
        target_train = None,
        source_val = None, 
        target_val = None,
        opt = {}
        ):

    wandb.init(project="robust-domain-adaptation")
    wandb.config = opt

    # Calculate output of image discriminator (PatchGAN)
    patch = int(opt["img_size"] / 2 ** 4)
    patch = (1, patch, patch)

    wandb.config = opt

    # Loss function
    adversarial_loss = torch.nn.MSELoss()
    task_loss = torch.nn.CrossEntropyLoss()

    # Loss weights
    lambda_adv = 1
    lambda_task = 0.1

    if cuda:
        generator.cuda()
        discriminator.cuda()
        classifier.cuda()
        adversarial_loss.cuda()
        task_loss.cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    classifier.apply(weights_init_normal)

    # Configure data loader
    dataloader_A = source_train
    dataloader_A_test = source_val

    dataloader_B = target_train
    dataloader_B_test = target_val

    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(generator.parameters(), classifier.parameters()), lr=opt["lr"], betas=(opt["b1"], opt["b2"])
    )
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt["lr"], betas=(opt["b1"], opt["b2"]))

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    # ----------
    #  Training
    # ----------

    # Keeps 100 accuracy measurements
    task_performance = []
    target_performance = []

    for epoch in range(opt["num_epochs"]):

        for i, ((imgs_A, labels_A), (imgs_B, labels_B)) in enumerate(zip(dataloader_A, dataloader_B)):

            batch_size = imgs_A.size(0)

            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, *patch).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, *patch).fill_(0.0), requires_grad=False)

            # Configure input
            imgs_A = Variable(imgs_A.type(FloatTensor).expand(batch_size, 3, opt["img_size"], opt["img_size"]))
            labels_A = Variable(labels_A.type(LongTensor))
            imgs_B = Variable(imgs_B.type(FloatTensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise
            z = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, opt["latent_dim"]))))

            # Generate a batch of images
            fake_B = generator(imgs_A, z)

            # Perform task on translated source image
            label_pred = classifier(fake_B)

            # Calculate the task loss
            task_loss_ = (task_loss(label_pred, labels_A) + task_loss(classifier(imgs_A), labels_A)) / 2

            # Loss measures generator's ability to fool the discriminator
            g_loss = lambda_adv * adversarial_loss(discriminator(fake_B), valid) + lambda_task * task_loss_

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(imgs_B), valid)
            fake_loss = adversarial_loss(discriminator(fake_B.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # ---------------------------------------
            #  Evaluate Performance on target domain
            # ---------------------------------------

            # Evaluate performance on translated Domain A
            acc = np.mean(np.argmax(label_pred.data.cpu().numpy(), axis=1) == labels_A.data.cpu().numpy())
            task_performance.append(acc)
            if len(task_performance) > 100:
                task_performance.pop(0)

            # Evaluate performance on Domain B
            pred_B = classifier(imgs_B)
            target_acc = np.mean(np.argmax(pred_B.data.cpu().numpy(), axis=1) == labels_B.numpy())
            target_performance.append(target_acc)
            if len(target_performance) > 100:
                target_performance.pop(0)

            # Evaluate performance in the original Domain A 

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [CLF acc: %3d%% (%3d%%), target_acc: %3d%% (%3d%%)]"
                % (
                    epoch,
                    opt["num_epochs"],
                    i,
                    len(dataloader_A),
                    d_loss.item(),
                    g_loss.item(),
                    100 * acc,
                    100 * np.mean(task_performance),
                    100 * target_acc,
                    100 * np.mean(target_performance),
                )
            )
        
        global_acc_a, confusion_a = validate(dataloader_A_test, classifier, opt, device)
        global_acc_b, confusion_b = validate(dataloader_B_test, classifier, opt, device)

        per_class_a = np.diag(confusion_a).tolist()
        per_class_b = np.diag(confusion_b).tolist()

        logged_metrics = [ ("validation accuracy (target)", global_acc_b), ("validation accuracy (source)", global_acc_a) ]
        logged_metrics += [ (pair[0] + "_target", pair[1]) for pair in zip(opt["class_names"], per_class_b) ]
        logged_metrics += [ (pair[0] + "_source", pair[1]) for pair in zip(opt["class_names"], per_class_a) ]

        log_ = { pair[0] : pair[1] for pair in logged_metrics }

        wandb.log(log_)

    global_acc_a, confusion_a = validate(dataloader_A_test, classifier, opt, device)
    global_acc_b, confusion_b = validate(dataloader_B_test, classifier, opt, device)

    f_a = plotly_confusion_matrix(confusion_a, opt["class_names"])
    f_b = plotly_confusion_matrix(confusion_b, opt["class_names"])

    wandb.log({"source confusion": wandb.Plotly(f_a)})
    wandb.log({"target confusion": wandb.Plotly(f_b)})

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

    @author: Junguang Jiang
    @contact: JiangJunguang1123@outlook.com
    """

    wandb.init(project="robust-domain-adaptation")
    wandb.config = opt

    classifier.train()

    train_source_iter = ForeverDataIterator(source_train)
    train_target_iter = ForeverDataIterator(target_train)

    classifier = classifier.to(device)

    # define optimizer and lr scheduler
    optimizer = SGD(classifier.get_parameters(), opt["lr"], momentum=opt["momentum"], weight_decay=opt["weight_decay"],nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: opt["lr"] * (1. + opt["lr_gamma"] * float(x)) ** (-opt["lr_decay"]))
    mcc_loss = MinimumClassConfusionLoss(temperature=opt["temperature"])

    for e in range(opt["epochs"]):
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
    
    @author: Junguang Jiang
    @contact: JiangJunguang1123@outlook.com
    """

    wandb.init(project="robust-domain-adaptation")
    wandb.config = opt

    classifier.train()

    train_source_iter = ForeverDataIterator(source_train)
    train_target_iter = ForeverDataIterator(target_train)

    classifier = classifier.to(device)

    # define optimizer and lr scheduler
    mdd = MarginDisparityDiscrepancy(opt["margin"]).to(device)
    optimizer = SGD(classifier.get_parameters(), opt["lr"], momentum=opt["momentum"], weight_decay=opt["weight_decay"], nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: opt["lr"] * (1. + opt["lr_gamma"] * float(x)) ** (-opt["lr_decay"]))

    for e in range(opt["epochs"]):
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
            classifier.step()

            # compute gradient and do SGD step
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

