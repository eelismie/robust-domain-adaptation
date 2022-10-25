import argparse
import os
import numpy as np
import math
import itertools

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from torchmetrics import ConfusionMatrix, Accuracy

import torch.nn as nn
import torch.nn.functional as F
import torch

import wandb

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def validate(val_loader, model, args, device):

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
    return acc.cpu().numpy(), torch.diag(confusion, 0).cpu().numpy().tolist()

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

    # Calculate output of image discriminator (PatchGAN)
    patch = int(opt["img_size"] / 2 ** 4)
    patch = (1, patch, patch)

    wandb.config = {
        "learning_rate": opt["lr"],
        "epochs": opt["num_epochs"]
    }

    cuda = True if torch.cuda.is_available() else False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        
        global_acc_a, per_class_a = validate(dataloader_A_test, classifier, opt, device)
        global_acc_b, per_class_b = validate(dataloader_B_test, classifier, opt, device)

        logged_metrics = [("validation accuracy (target)", global_acc_b), ("validation accuracy (source)", global_acc_a)]
        logged_metrics += [ (pair[0] + "_target", pair[1]) for pair in zip(opt["class_names"], per_class_b) ]
        logged_metrics += [ (pair[0] + "_source", pair[1]) for pair in zip(opt["class_names"], per_class_a) ]

        log_ = { pair[0] : pair[1] for pair in logged_metrics }

        wandb.log(log_)
