import numpy as np
import os
import plotly.express as px 

from torchattacks import PGD
from torch.utils.data import DataLoader
from dataset_registry.datasets import ClassSampler
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from tllib.vision.datasets import ImageList
from tllib.utils.data import ForeverDataIterator
from tllib.alignment.mdd import ClassificationMarginDisparityDiscrepancy as MarginDisparityDiscrepancy
from tllib.alignment.cdan import ConditionalDomainAdversarialLoss
from tllib.modules.domain_discriminator import DomainDiscriminator
from tllib.self_training.mcc import MinimumClassConfusionLoss

from torchmetrics import ConfusionMatrix, Accuracy
from torch.utils.data.sampler import RandomSampler, SequentialSampler
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

def compute_class_indices(target_train_loader, classifier, opt):

    """
    Make one pass through the target train dataset, and perform the following steps: 
    1. Compute a classifier prediction of the target domain labels
    2. Place the corresponding target images into sampling buckets (clustering step) based on predicted class 
    3. During training, draw one of these buckets at random and form batches based on these images.
    """

    #load datapoints sequentially
    seq_loader = DataLoader(target_train_loader.dataset, batch_size=opt["batch_size"], drop_last=False, shuffle=False)
    all_pseudo_targets = []

    classifier.eval()

    try:
        #DEBUGGING
        #when switching to cfol sampling for the first time, this should throw an exeption
        #beacuse the class indices aren't defined.
        print(target_train_loader.dataset.class_indices[:10])
    except Exception as e:
        print(e) 

    with torch.no_grad():
        for i, data in tqdm(enumerate(seq_loader)):
            images, labels = data #disregard labels, compute classifier predictions
            images = images.to(device)
            outputs = classifier(images)                 
            pseudo_labels = torch.argmax(outputs, 1).cpu().numpy().flatten().tolist()
            all_pseudo_targets = all_pseudo_targets + pseudo_labels

    all_pseudo_targets = np.array(all_pseudo_targets)

    #set class_indices in the dataset to the predicted ones
    target_train_loader.dataset.class_indices = [
        (all_pseudo_targets == class_id).nonzero()[0] for class_id in range(target_train_loader.dataset.num_classes)
        ]

    classifier.train()

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

        self.classifier = classifier.model 
        self.source_train = source_train
        self.target_train = target_train
        self.source_val = source_val
        self.target_val = DataLoader(self.target_train.dataset, batch_size=opt["batch_size"], shuffle=False)
        self.target_test = target_val 
        self.num_classes = target_train.dataset.num_classes

        #                ___A note on UDA validation / test split___
        # - the target train dataset labels are used for validation (assuming no target labels are used in training the model) 
        # - the target test split is used for testing (data seen by model in production)

        assert(isinstance(self.target_train.sampler, RandomSampler))
        assert(isinstance(self.target_train.sampler, RandomSampler))  
        assert(isinstance(self.target_val.sampler, SequentialSampler))
        assert(isinstance(self.target_test.sampler, SequentialSampler)) 

        self.opt = opt

    def run(self):
        self.setup()
        for e in range(self.opt["num_epochs"]):
            self.each_epoch(e)
        self.cleanup()

    def each_epoch(self, e):

        for i in range(self.opt["iters_per_epoch"]):
            self.each_iter(e,i) 

        global_acc_a, confusion_a = validate(self.source_val, self.classifier, self.opt)
        global_acc_b, confusion_b = validate(self.target_val, self.classifier, self.opt)

        per_class_a = np.diag(confusion_a).tolist()
        per_class_b = np.diag(confusion_b).tolist()

        logged_metrics = [ ("validation accuracy (target)", global_acc_b), ("validation accuracy (source)", global_acc_a) ]
        logged_metrics += [ (pair[0] + "_target", pair[1]) for pair in zip(self.opt["class_names"], per_class_b) ]
        logged_metrics += [ (pair[0] + "_source", pair[1]) for pair in zip(self.opt["class_names"], per_class_a) ]

        log_ = { pair[0] : pair[1] for pair in logged_metrics }

        wandb.log(log_)

    def setup(self):
        set_global_seed(self.opt)
        wandb.init(project="robust-domain-adaptation", config=self.opt)
        self.classifier.train()
        self.classifier = self.classifier.to(device)
        self.optimizer = SGD(self.classifier.get_parameters(), self.opt["lr"], momentum=self.opt["momentum"], weight_decay=self.opt["weight_decay"],nesterov=True)
        self.lr_scheduler = LambdaLR(self.optimizer, lambda x: self.opt["lr"] * (1. + self.opt["lr_gamma"] * float(x)) ** (-self.opt["lr_decay"]))

    def cleanup(self):

        global_acc_a, confusion_a = validate(self.source_val, self.classifier, self.opt)
        global_acc_b, confusion_b = validate(self.target_val, self.classifier, self.opt)
        global_acc_c, confusion_c = validate(self.target_test, self.classifier, self.opt)

        f_a = plotly_confusion_matrix(confusion_a, self.opt["class_names"])
        f_b = plotly_confusion_matrix(confusion_b, self.opt["class_names"])
        f_c = plotly_confusion_matrix(confusion_c, self.opt["class_names"])

        wandb.log({"source confusion": wandb.Plotly(f_a)})
        wandb.log({"target confusion": wandb.Plotly(f_b)}) 
        wandb.log({"target confusion (test)": wandb.Plotly(f_c)}) 
        wandb.log({"source validation accuracy": global_acc_a})
        wandb.log({"target validation accuracy": global_acc_b})
        wandb.log({"target test accuracy": global_acc_c})

        return

    def each_iter(self, e, i):
        raise NotImplementedError

    def insertClassIndices(self, dataset: ImageList) -> ImageList:
        # required for a dataset to be compatible with cfol sampler
        if not isinstance(dataset.targets, np.ndarray):
            dataset.targets = np.array(dataset.targets)
        dataset.class_indices = [(dataset.targets == class_id).nonzero()[0] for class_id in range(dataset.num_classes)]
        return dataset

    def switch_to_cfol(self):
        compute_class_indices(self.target_train, self.classifier, self.opt)
        source_train_sampler = ClassSampler(self.insertClassIndices(self.source_train.dataset), gamma=self.opt["cfol_gamma"])
        target_train_sampler = ClassSampler(self.target_train.dataset)
        self.source_train = DataLoader(self.source_train.dataset, batch_size=self.opt["batch_size"], sampler=source_train_sampler, drop_last=True)
        self.target_train = DataLoader(self.target_train.dataset, batch_size=self.opt["batch_size"], sampler=target_train_sampler, drop_last=True)
        self.train_source_iter = ForeverDataIterator(self.source_train)
        self.train_target_iter = ForeverDataIterator(self.target_train)
        self.sampler = self.source_train.sampler

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

    def each_iter(self, e, i):
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
        transfer_loss = self.mcc_loss(y_t)
        loss = cls_loss + transfer_loss * self.opt["trade_off"]

        if (i % 10 == 0):
            print("cls loss: ", cls_loss.item(), " trans loss: ", transfer_loss.item(), " total loss: ", loss.item())

        # compute gradient and do SGD step
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

    def cleanup(self):
        super().cleanup()
        now = datetime.now()
        current_time = now.strftime("%D:%H:%M:%S").replace("/","").replace(":","")
        dataset = type(self.source_train.dataset).__name__
        model_path = current_time + dataset + "_mcc.pt"
        savedir = os.path.join(self.opt["checkpoint_path"], model_path)
        torch.save(self.classifier.state_dict(), savedir)
        return


class cdan_experiment(experiment):

    def setup(self):

        wandb.init(project="robust-domain-adaptation", config=self.opt)
        
        set_global_seed(self.opt)

        self.train_source_iter = ForeverDataIterator(self.source_train)
        self.train_target_iter = ForeverDataIterator(self.target_train)
        self.classifier.train()
        self.classifier = self.classifier.to(device)

        classifier_feature_dim = self.classifier.features_dim
        domain_discri = DomainDiscriminator(classifier_feature_dim * self.num_classes, hidden_size=1024).to(device)

        all_parameters = self.classifier.get_parameters() + domain_discri.get_parameters()
        self.optimizer = SGD(all_parameters, self.opt["lr"], momentum=self.opt["momentum"],weight_decay=self.opt["weight_decay"], nesterov=True)
        self.lr_scheduler = LambdaLR(self.optimizer, lambda x: self.opt["lr"] * (1. + self.opt["lr_gamma"] * float(x)) ** (-self.opt["lr_decay"]))
        self.domain_adv = ConditionalDomainAdversarialLoss(
            domain_discri, entropy_conditioning=False,
            num_classes=self.num_classes, features_dim=classifier_feature_dim, randomized=False,
            randomized_dim=1024
        ).to(device)
        self.domain_adv.train()

    def each_iter(self, e, i):
        x_s, labels_s = next(self.train_source_iter)[:2]
        x_t, = next(self.train_target_iter)[:1]

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)

        # compute output
        x = torch.cat((x_s, x_t), dim=0)
        y, f = self.classifier(x)
        y_s, y_t = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)

        cls_loss = F.cross_entropy(y_s, labels_s)
        transfer_loss = self.domain_adv(y_s, f_s, y_t, f_t)
        loss = cls_loss + transfer_loss * self.opt["trade_off"]

        if (i % 10 == 0):
            print("cls loss: ", cls_loss.item(), " trans loss: ", transfer_loss.item(), " total loss: ", loss.item())

        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()    

class cdan_experiment_cfol(cdan_experiment):

    def each_epoch(self, e):

        #reinitialize all samplers 
        if (e == self.opt["cfol_epoch"]):
            self.switch_to_cfol()

        for i in range(self.opt["iters_per_epoch"]):
            if (e >= self.opt["cfol_epoch"]):
                self.each_iter_cfol(e, i)
            else:
                self.each_iter(e,i) 

        global_acc_a, confusion_a = validate(self.source_val, self.classifier, self.opt)
        global_acc_b, confusion_b = validate(self.target_val, self.classifier, self.opt)

        per_class_a = np.diag(confusion_a).tolist()
        per_class_b = np.diag(confusion_b).tolist()

        logged_metrics = [ ("validation accuracy (target)", global_acc_b), ("validation accuracy (source)", global_acc_a) ]
        logged_metrics += [ (pair[0] + "_target", pair[1]) for pair in zip(self.opt["class_names"], per_class_b) ]
        logged_metrics += [ (pair[0] + "_source", pair[1]) for pair in zip(self.opt["class_names"], per_class_a) ]

        log_ = { pair[0] : pair[1] for pair in logged_metrics }

        wandb.log(log_)

    def each_iter_cfol(self, e, i):
        x_s, labels_s = next(self.train_source_iter)[:2]
        x_t, = next(self.train_target_iter)[:1]

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)

        # compute output
        x = torch.cat((x_s, x_t), dim=0)
        y, f = self.classifier(x)
        y_s, y_t = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)

        cls_loss = F.cross_entropy(y_s, labels_s, reduction="none")
        transfer_loss = self.domain_adv(y_s, f_s, y_t, f_t)

        if self.sampler.reweight:
            cls_loss = self.num_classes * self.sampler.batch_weight(labels_s).type_as(cls_loss) * cls_loss

        loss = cls_loss.mean() + transfer_loss * self.opt["trade_off"]

        class_sampler_lr = 0.0000001
        predictions = torch.argmax(y_s, 1)
        class_loss = predictions != labels_s
        eta_times_loss_arms = class_sampler_lr * class_loss
        self.sampler.batch_update(labels_s, eta_times_loss_arms)

        if (i % 10 == 0):
            print("cls loss: ", cls_loss.mean().item(), " trans loss: ", transfer_loss.item(), " total loss: ", loss.item())

        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step() 
    
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
        self.mdd.train()

    def each_iter(self, e, i):
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
        self.classifier.step()

        if (i % 10 == 0):
            print("cls loss: ", cls_loss.item(), " trans loss: ", transfer_loss.item(), " total loss: ", loss.item())

        # compute gradient and do SGD step
        loss.backward() 
        self.optimizer.step()
        self.lr_scheduler.step()

    def cleanup(self):
        super().cleanup()
        now = datetime.now()
        current_time = now.strftime("%D:%H:%M:%S").replace("/","").replace(":","")
        dataset = type(self.source_train.dataset).__name__
        model_path = current_time + dataset + "_mdd.pt"
        savedir = os.path.join(self.opt["checkpoint_path"], model_path)
        torch.save(self.classifier.state_dict(), savedir)
        return

class mcc_experiment_cfol(mcc_experiment):

    def __init__(self, classifier=None, source_train=None, target_train=None, source_val=None, target_val=None, opt=None):
        super().__init__(classifier, source_train, target_train, source_val, target_val, opt)

    def each_epoch(self, e):

        #reinitialize all samplers 
        if (e == self.opt["cfol_epoch"]):
            self.switch_to_cfol()

        for i in range(self.opt["iters_per_epoch"]):
            if (e >= self.opt["cfol_epoch"]):
                self.each_iter_cfol(e, i)
            else:
                self.each_iter(e,i) 

        global_acc_a, confusion_a = validate(self.source_val, self.classifier, self.opt)
        global_acc_b, confusion_b = validate(self.target_val, self.classifier, self.opt)

        per_class_a = np.diag(confusion_a).tolist()
        per_class_b = np.diag(confusion_b).tolist()

        logged_metrics = [ ("validation accuracy (target)", global_acc_b), ("validation accuracy (source)", global_acc_a) ]
        logged_metrics += [ (pair[0] + "_target", pair[1]) for pair in zip(self.opt["class_names"], per_class_b) ]
        logged_metrics += [ (pair[0] + "_source", pair[1]) for pair in zip(self.opt["class_names"], per_class_a) ]

        log_ = { pair[0] : pair[1] for pair in logged_metrics }

        wandb.log(log_)

    def each_iter_cfol(self, e, i):
        
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

        cls_loss = F.cross_entropy(y_s, labels_s, reduction="none")
        transfer_loss = self.mcc_loss(y_t)

        # Possibly weight losses
        if self.sampler.reweight:
            cls_loss = self.num_classes * self.sampler.batch_weight(labels_s).type_as(cls_loss) * cls_loss

        loss = cls_loss.mean() + transfer_loss * self.opt["trade_off"]

        class_sampler_lr = 0.0000001
        predictions = torch.argmax(y_s, 1)
        class_loss = predictions != labels_s
        eta_times_loss_arms = class_sampler_lr * class_loss
        self.sampler.batch_update(labels_s, eta_times_loss_arms)

        if (i % 10 == 0):
            print("cls loss: ", cls_loss.mean().item(), " trans loss: ", transfer_loss.item(), " total loss: ", loss.item())

        # compute gradient and do SGD step
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

class mdd_experiment_cfol(mdd_experiment):

    """
    Maximum distribution discrepancy with CFOL
    """

    def __init__(self, classifier=None, source_train=None, target_train=None, source_val=None, target_val=None, opt=None):
        super().__init__(classifier, source_train, target_train, source_val, target_val, opt)

    def each_epoch(self, e):

        #reinitialize all samplers 
        if (e == self.opt["cfol_epoch"]):
            self.switch_to_cfol()

        for i in range(self.opt["iters_per_epoch"]):
            if (e >= self.opt["cfol_epoch"]):
                self.each_iter_cfol(e, i)
            else:
                self.each_iter(e,i) 

        global_acc_a, confusion_a = validate(self.source_val, self.classifier, self.opt)
        global_acc_b, confusion_b = validate(self.target_val, self.classifier, self.opt)

        per_class_a = np.diag(confusion_a).tolist()
        per_class_b = np.diag(confusion_b).tolist()

        logged_metrics = [ ("validation accuracy (target)", global_acc_b), ("validation accuracy (source)", global_acc_a) ]
        logged_metrics += [ (pair[0] + "_target", pair[1]) for pair in zip(self.opt["class_names"], per_class_b) ]
        logged_metrics += [ (pair[0] + "_source", pair[1]) for pair in zip(self.opt["class_names"], per_class_a) ]

        log_ = { pair[0] : pair[1] for pair in logged_metrics }

        wandb.log(log_)

    def each_iter_cfol(self, e, i):
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
        cls_loss = F.cross_entropy(y_s, labels_s, reduction="none")
        transfer_loss = -self.mdd(y_s, y_s_adv, y_t, y_t_adv)

        if self.sampler.reweight:
            cls_loss = self.num_classes * self.sampler.batch_weight(labels_s).type_as(cls_loss) * cls_loss

        loss = cls_loss.mean() + transfer_loss * self.opt["trade_off"]
        self.classifier.step()

        class_sampler_lr = 0.0000001
        predictions = torch.argmax(y_s, 1)
        class_loss = predictions != labels_s
        eta_times_loss_arms = class_sampler_lr * class_loss
        self.sampler.batch_update(labels_s, eta_times_loss_arms)

        if (i % 10 == 0):
            print("cls loss: ", cls_loss.mean().item(), " trans loss: ", transfer_loss.item(), " total loss: ", loss.item())

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

    if (opt["cfol_sampling"]):
        experiment_ = mcc_experiment_cfol(
            classifier=classifier, 
            source_train=source_train, 
            target_train=target_train,
            source_val=source_val,
            target_val=target_val,
            opt=opt)
    else:
        experiment_ = mcc_experiment(
            classifier=classifier, 
            source_train=source_train, 
            target_train=target_train,
            source_val=source_val,
            target_val=target_val,
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

    if (opt["cfol_sampling"]):
        experiment_ = mdd_experiment_cfol(
            classifier=classifier, 
            source_train=source_train, 
            target_train=target_train,
            source_val=source_val,
            target_val=target_val,
            opt=opt)
    else:
        experiment_ = mdd_experiment(
            classifier=classifier, 
            source_train=source_train, 
            target_train=target_train,
            source_val=source_val,
            target_val=target_val,
            opt=opt)

    experiment_.run()
    return

def train_cdan(
    classifier = None,
    source_train = None,
    target_train = None,
    source_val = None,
    target_val = None,
    opt = None
    ):

    if (opt["cfol_sampling"]):
        experiment_ = cdan_experiment_cfol(
            classifier=classifier, 
            source_train=source_train, 
            target_train=target_train,
            source_val=source_val,
            target_val=target_val,
            opt=opt)
    else:
        experiment_ = cdan_experiment(
            classifier=classifier, 
            source_train=source_train, 
            target_train=target_train,
            source_val=source_val,
            target_val=target_val,
            opt=opt)

    experiment_.run()
    return  