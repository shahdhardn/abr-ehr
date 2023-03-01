import os
import time
import argparse
import random
import copy
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import shutil
from Sinkhorn_distance import SinkhornDistance
from Sinkhorn_distance_fl import SinkhornDistance as SinkhornDistance_fl
from torch.utils.data import TensorDataset, DataLoader
from sklearn import metrics
from data_module import *

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from models import *

from data_module import MimicDataModule
from data_utils import *

# Log in to your W&B account
import wandb

wandb.login()

parser = argparse.ArgumentParser(description="Imbalanced Example")
# TODO dataset
# parser.add_argument(
#     "--dataset",
#     default="cifar10",
#     type=str,
#     help="dataset (cifar10[default] or cifar100)",
# )
parser.add_argument(
    "--dataset",
    default="mimic",
    type=str,
    help="mimic",
)
parser.add_argument(
    "--cost", default="combined", type=str, help="[combined, label, feature, twoloss]"
)
# TODO meta set
# parser.add_argument(
#     "--meta_set", default="prototype", type=str, help="[whole, prototype]"
# )
parser.add_argument("--meta_set", default="whole", type=str, help="[whole, prototype]")
# TODO change batch size
parser.add_argument(
    "--batch-size",
    type=int,
    default=20,
    metavar="N",
    help="input batch size for training (default: 16)",
)
# TODO number of classes
# parser.add_argument("--num_classes", type=int, default=10)
parser.add_argument("--num_classes", type=int, default=2)
# TODO number of meta data
parser.add_argument(
    "--num_meta", type=int, default=10, help="The number of meta data for each class."
)
parser.add_argument("--imb_factor", type=float, default=0.08)
# TODO change number of epochs
# parser.add_argument(
#     "--epochs", type=int, default=250, metavar="N", help="number of epochs to train"
# )
parser.add_argument(
    "--epochs", type=int, default=14, metavar="N", help="number of epochs to train"
)
parser.add_argument(
    "--lr", "--learning-rate", default=1e-4, type=float, help="initial learning rate"
)
parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
parser.add_argument("--nesterov", default=True, type=bool, help="nesterov momentum")
parser.add_argument(
    "--weight-decay",
    "--wd",
    default=5e-4,
    type=float,
    help="weight decay (default: 5e-4)",
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)
# parser.add_argument(ß
#     "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
# )
parser.add_argument(
    "--print-freq", "-p", default=100, type=int, help="print frequency (default: 100)"
)
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument("--save_name", default="OT_ARF12_imb0.08", type=str)
parser.add_argument("--idx", default="ours", type=str)

parser.add_argument("--model", type=str, default="MBertLstm")
parser.add_argument(
    "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
)

args = parser.parse_args()
for arg in vars(args):
    print("{}={}".format(arg, getattr(args, arg)))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
# kwargs = {"num_workers": 16, "pin_memory": False}
use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")

mimic_data_module = MimicDataModule.from_argparse_args(args)
mimic_data_module.prepare_data()
mimic_data_module.setup()
imbalanced_train_loader = mimic_data_module.train_dataloader()
# validation_loader = mimic_data_module.val_dataloader()
test_loader = mimic_data_module.test_dataloader()

imbalanced_train_labels = imbalanced_train_loader.dataset.y

class_counts = [
    len(imbalanced_train_labels[imbalanced_train_labels == i])
    for i in range(args.num_classes)
]

meta_loader = mimic_data_module.meta_dataloader()

print("Numper of samples per class: ", class_counts)  # to calculate imbalance factor
args.imb_factor = (np.max(class_counts) / np.min(class_counts)) / 100
print("Imbalance factor: ", args.imb_factor)

print("train data size: ", len(imbalanced_train_loader.dataset.y))
# print("validation data size: ", len(validation_loader.dataset.y))
print("meta data size: ", len(meta_loader.dataset.y))
print("test data size: ", len(test_loader.dataset.y))

print("train batches= ", len(imbalanced_train_loader), " batches")
# print("validation batches= ", len(validation_loader), " batches")
print("meta batches= ", len(meta_loader), " batches")
print("test batches= ", len(test_loader), " batches")

# count number of samples per class in meta set to make sure it is balanced
meta_data, meta_labels, _ = next(iter(meta_loader))
meta_class_counts = [
    len(meta_labels[meta_labels == i]) for i in range(args.num_classes)
]
print("meta_class_counts: ", meta_class_counts)
del meta_data, meta_labels, meta_class_counts

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
classe_labels = range(args.num_classes)

# samples_num_list = get_samples_num_per_cls(
#     args.dataset,
#     args.imb_factor,
#     args.num_meta * args.num_classes,
#     len(imbalanced_train_loader.dataset.y),
# )
# print(samples_num_list)
# print(sum(samples_num_list))

best_prec1 = 0
best_f1 = 0
best_auroc = 0
best_aupr = 0

# Assigning weights to each datapoint
beta = 0.9999
effective_num = 1.0 - np.power(beta, class_counts)
per_cls_weights = (1.0 - beta) / np.array(effective_num)
per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(class_counts)
per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
weights = torch.tensor(per_cls_weights).float()
weightsbuffer = torch.tensor(
    [per_cls_weights[cls_i] for cls_i in imbalanced_train_labels]
).to(device)

# breakpoint()

eplisons = 0.1
criterion = SinkhornDistance(eps=eplisons, max_iter=200, reduction=None, dis="cos").to(
    device
)
criterion_label = SinkhornDistance(
    eps=eplisons, max_iter=200, reduction=None, dis="euc"
).to(device)
criterion_fl = SinkhornDistance_fl(eps=eplisons, max_iter=200, reduction=None).to(
    device
)

run = wandb.util.generate_id()
wandb.init(
    # Set the project where this run will be logged
    project="OT_ARF12",
    # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    name=f"experiment_{run}",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": args.lr,
        "architecture": "MBertLSTM",
        "dataset": "MIMIC",
        "epochs": args.epochs,
        "imb_factor": args.imb_factor,
        "batch_size": args.batch_size,
    },
)


def main():
    # TODO checkpoints
    global args, best_prec1
    args = parser.parse_args()

    if args.dataset == "cifar10":
        if args.imb_factor == 0.005:
            ckpt_path = r"checkpoint/ours/pretrain/.."

    else:
        if args.imb_factor == 0.005:
            ckpt_path = r"checkpoint/ours/pretrain/.."
        else:
            ckpt_path = r"checkpoint/ours/pretrain/.."

    # TODO True
    model = build_model(load_pretrain=False, ckpt_path=ckpt_path)
    optimizer_a = torch.optim.SGD(
        [model.linear.weight, model.linear.bias],
        args.lr,
        momentum=args.momentum,
        nesterov=args.nesterov,
        weight_decay=args.weight_decay,
    )
    # optimizer_a = torch.optim.AdamW(
    #     [model.linear.weight, model.linear.bias],
    #     args.lr,
    #     weight_decay=args.weight_decay,
    # )
    cudnn.benchmark = True
    criterion_classifier = nn.CrossEntropyLoss(reduction="none").to(device)

    for epoch in range(0, args.epochs):

        train_OT(
            imbalanced_train_loader,
            meta_loader,
            weightsbuffer,
            model,
            optimizer_a,
            epoch,
            criterion_classifier,
        )

        prec1, f1, auroc, aupr, preds, gt_labels = validate(test_loader, model)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if is_best:
            weightsbuffer_bycls = []
            for i_cls in range(args.num_classes):
                weightsbuffer_bycls.extend(
                    weightsbuffer[imbalanced_train_labels == i_cls].data.cpu().numpy()
                )
            corresponding_f1 = f1
            corresponding_auroc = auroc
            corresponding_aupr = aupr

        save_checkpoint(
            args,
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_acc1": best_prec1,
                "corresponding_f1": corresponding_f1,
                "corresponding_auroc": corresponding_auroc,
                "corresponding_aupr": corresponding_aupr,
                "optimizer": optimizer_a.state_dict(),
                "weights": weightsbuffer_bycls,
            },
            is_best,
        )

    print(
        "Best accuracy: ",
        best_prec1,
        "Corresponding F1: ",
        corresponding_f1,
        "Corresponding AUROC: ",
        corresponding_auroc,
        "Corresponding AUPR: ",
        corresponding_aupr,
    )

    wandb.finish()


def train_OT(
    train_loader,
    validation_loader,
    weightsbuffer,
    model,
    optimizer,
    epoch,
    criterion_classifier,
):
    losses = AverageMeter()
    otlosses = AverageMeter()
    train_losses = AverageMeter()
    val_losses = AverageMeter()
    top1 = AverageMeter()
    f1 = AverageMeter()
    auroc = AverageMeter()
    aupr = AverageMeter()

    model.train()

    val_data, val_labels, _ = next(iter(validation_loader))
    print(np.unique(val_labels.cpu(), return_counts=True))
    val_data[0][0], val_data[0][1], val_data[1], val_data[2], val_data[3] = to_var_x(
        val_data, requires_grad=False
    )
    val_labels = to_var(val_labels, requires_grad=False).squeeze()

    # breakpoint()

    if args.meta_set == "whole":
        val_data_bycls = val_data
        val_labels_bycls = val_labels
    elif args.meta_set == "prototype":
        val_data_bycls = torch.zeros([args.num_classes, args.num_meta, 3, 32, 32]).to(
            device
        )
        for i_cls in range(args.num_classes):
            val_data_bycls[i_cls, ::] = val_data[val_labels == i_cls]
        val_data_bycls = torch.mean(val_data_bycls, dim=1)
        val_labels_bycls = torch.tensor([i_l for i_l in range(args.num_classes)]).to(
            device
        )

    val_labels_onehot = to_categorical(val_labels_bycls).to(device)
    feature_val, _ = model(val_data_bycls)
    for i, batch in enumerate(train_loader):
        inputs, labels, ids = tuple(t for t in batch)
        inputs[0][0], inputs[0][1], inputs[1], inputs[2], inputs[3] = to_var_x(
            inputs, requires_grad=False
        )

        labels = labels.to(device)
        ids = ids.to(device)
        labels = labels.squeeze()
        labels_onehot = to_categorical(labels).to(device)

        # breakpoint()

        weights = to_var(weightsbuffer[ids])
        model.eval()
        Attoptimizer = torch.optim.SGD(
            [weights], lr=0.01, momentum=0.9, weight_decay=5e-4
        )
        # Attoptimizer = torch.optim.AdamW(
        #     [weights], lr=args.lr, weight_decay=args.weight_decay
        # )
        for ot_epoch in range(1):
            feature_train, _ = model(inputs)
            probability_train = softmax_normalize(weights)

            if args.cost == "feature":
                OTloss = criterion(
                    feature_val.detach(),
                    feature_train.detach(),
                    probability_train.squeeze(),
                )
            elif args.cost == "label":
                OTloss = criterion_label(
                    torch.tensor(val_labels_onehot, dtype=float).to(device),
                    torch.tensor(labels_onehot, dtype=float).to(device),
                    probability_train.squeeze(),
                )
            elif args.cost == "combined":
                OTloss = criterion_fl(
                    feature_val.detach(),
                    feature_train.detach(),
                    torch.tensor(val_labels_onehot, dtype=float).to(device),
                    torch.tensor(labels_onehot, dtype=float).to(device),
                    probability_train.squeeze(),
                )
            elif args.cost == "twoloss":
                OTloss1 = criterion(
                    feature_val.detach(),
                    feature_train.detach(),
                    probability_train.squeeze(),
                )
                OTloss2 = criterion_label(
                    torch.tensor(val_labels_onehot, dtype=float).to(device),
                    torch.tensor(labels_onehot, dtype=float).to(device),
                    probability_train.squeeze(),
                )
                OTloss = OTloss1 + OTloss2

            Attoptimizer.zero_grad()
            OTloss.backward(retain_graph=False)
            Attoptimizer.step()

        weightsbuffer[ids] = weights.data

        # breakpoint()

        model.train()
        optimizer.zero_grad()
        _, logits = model(inputs)
        loss_train = criterion_classifier(logits, labels.long())
        _, logits_val = model(val_data)
        loss_val = F.cross_entropy(logits_val, val_labels.long(), reduction="none")
        loss = torch.sum(loss_train * weights.data) + 10 * torch.mean(loss_val)
        loss.backward(retain_graph=False)
        optimizer.step()

        # breakpoint()

        prec_train = accuracy(logits.data, labels, topk=(1,))[0]
        f1_acc = calc_f1(logits.data, labels)[0]
        try:
            au_roc = calc_auroc(logits.data, labels)[0]
        except Exception as ex:
            print("AUROC_ERROR in batch number: ", i, ex.__class__.__name__)
            au_roc = [0.0]
        au_pr = calc_aupr(logits.data, labels)[0]

        otlosses.update(OTloss.item(), labels.size(0))
        losses.update(loss.item(), labels.size(0))
        train_losses.update(loss_train.mean().item(), labels.size(0))
        val_losses.update(loss_val.mean().item(), labels.size(0))
        top1.update(prec_train.item(), labels.size(0))
        f1.update(f1_acc.item(), labels.size(0))
        auroc.update(au_roc, labels.size(0))
        aupr.update(au_pr.item(), labels.size(0))
        if i == len(train_loader) - 1 or i % args.print_freq == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "F1 {f1.val:.3f} ({f1.avg:.3f})\t"
                "AUROC {auroc.val:.3f} ({auroc.avg:.3f})\t"
                "AUPR {aupr.val:.3f} ({aupr.avg:.3f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                    epoch,
                    i,
                    len(train_loader),
                    loss=losses,
                    f1=f1,
                    auroc=auroc,
                    aupr=aupr,
                    top1=top1,
                )
            )
    wandb.log(
        {
            "train-otloss": otlosses.avg,
            "loss": losses.avg,
            "train-loss": train_losses.avg,
            "val-loss": val_losses.avg,
            "train-prec1": top1.avg,
            "train-f1": f1.avg,
            "train-auroc": auroc.avg,
            "train-aupr": aupr.avg,
        }
    )


def validate(test_loader, model):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    f1 = AverageMeter()
    auroc = AverageMeter()
    aupr = AverageMeter()

    model.eval()

    true_labels = []
    preds = []

    end = time.time()
    for i, batch in enumerate(test_loader):
        input, target, _ = tuple(t for t in batch)
        target = target.to(device)
        input[0][0], input[0][1], input[1], input[2], input[3] = to_var_x(
            input, requires_grad=False
        )

        # # input_var = torch.autograd.Variable(input)
        # input_var[0][0] = torch.autograd.Variable(input_var[0][0])
        # input_var[0][1] = torch.autograd.Variable(input_var[0][1])
        # input_var[1] = torch.autograd.Variable(input_var[1])
        # input_var[2] = torch.autograd.Variable(input_var[2])
        # input_var[3] = torch.autograd.Variable(input_var[3])

        # target_var = torch.autograd.Variable(target)

        with torch.no_grad():
            _, output = model(input)

        output_numpy = output.data.cpu().numpy()
        preds_output = list(output_numpy.argmax(axis=1))

        true_labels += list(target.data.cpu().numpy())
        preds += preds_output

        targets_var = to_var(target, requires_grad=False).squeeze()
        loss = torch.tensor(F.cross_entropy(output.data, targets_var, reduction="none"))
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        f1_acc = calc_f1(output.data, target)[0]
        try:
            au_roc = calc_auroc(output.data, target)[0]
        except Exception as ex:
            print("AUROC_ERROR in batch number: ", i, ex.__class__.__name__)
            au_roc = [0.0]
        au_pr = calc_aupr(output.data, target)[0]

        losses.update(torch.mean(loss), target.size(0))
        top1.update(prec1.item(), target.size(0))
        f1.update(f1_acc.item(), target.size(0))
        auroc.update(au_roc, target.size(0))
        aupr.update(au_pr.item(), target.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i == len(test_loader) - 1:  # i % args.print_freq == 0:
            print(
                "Test: [{0}/{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "F1 {f1.val:.3f} ({f1.avg:.3f})\t"
                "AUROC {auroc.val:.3f} ({auroc.avg:.3f})\t"
                "AUPR {aupr.val:.3f} ({aupr.avg:.3f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                    i,
                    len(test_loader),
                    batch_time=batch_time,
                    loss=losses,
                    f1=f1,
                    auroc=auroc,
                    aupr=aupr,
                    top1=top1,
                )
            )

    print(" * Prec@1 {top1.avg:.3f}".format(top1=top1))
    print(" * F1 {f1.avg:.3f}".format(f1=f1))
    print(" * AUROC {auroc.avg:.3f}".format(auroc=auroc))
    print(" * AUPR {aupr.avg:.3f}".format(aupr=aupr))
    wandb.log(
        {
            "test-loss": losses.avg,
            "test-prec1": top1.avg,
            "test-f1": f1.avg,
            "test-auroc": auroc.avg,
            "test-aupr": aupr.avg,
        }
    )

    return top1.avg, f1.avg, auroc.avg, aupr.avg, preds, true_labels


# It's a wrapper around a DataParallel model that allows you to access the attributes of the
# underlying model
class _CustomDataParallel(nn.Module):
    def __init__(self, model):
        super(_CustomDataParallel, self).__init__()
        self.model = nn.DataParallel(model).cuda()

    def forward(self, *input):
        return self.model(*input)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model.module, name)


def build_model(load_pretrain, ckpt_path=None):
    """
    :param load_pretrain: whether to load pretrained model or not
    :param ckpt_path: the path to the checkpoint file
    :return: The model is being returned.
    """

    if args.model == "MBertLstm":
        if torch.cuda.is_available():
            model = _CustomDataParallel(MBertLstm()).to(device)
            torch.backends.cudnn.benchmark = True
        else:
            model = MBertLstm()

    elif args.model == "MBertCnn":
        if torch.cuda.is_available():
            model = _CustomDataParallel(MBertCnn()).to(device)
            torch.backends.cudnn.benchmark = True
        else:
            model = MBertCnn()

    if load_pretrain == True:
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint["model_state_dict"])

    return model


def to_var(y, requires_grad=True):
    if torch.cuda.is_available():
        y = y.to(device)
    return Variable(y, requires_grad=requires_grad)


def to_var_x(x, requires_grad=True):
    ti = torch.tensor(x[0][0])
    ts = torch.tensor(x[0][1])
    bert_sent = torch.tensor(x[1])
    bert_sent_type = torch.tensor(x[2])
    bert_sent_mask = torch.tensor(x[3])

    if torch.cuda.is_available():
        ti = ti.to(device)
        ts = ts.to(device)
        bert_sent = bert_sent.to(device)
        bert_sent_type = bert_sent_type.to(device)
        bert_sent_mask = bert_sent_mask.to(device)

    if requires_grad:
        return (
            torch.autograd.Variable(ti),
            torch.autograd.Variable(ts),
            torch.autograd.Variable(bert_sent),
            torch.autograd.Variable(bert_sent_type),
            torch.autograd.Variable(bert_sent_mask),
        )
    else:
        return (
            Variable(ti, requires_grad=requires_grad),
            Variable(ts, requires_grad=requires_grad),
            Variable(bert_sent, requires_grad=False),
            Variable(bert_sent_type, requires_grad=False),
            Variable(bert_sent_mask, requires_grad=False),
        )


def linear_normalize(weights):
    """
    It takes a vector of weights and returns a vector of weights that sums to 1

    :param weights: the weights of the different samples in the batch
    :return: The weights are being normalized to sum to 1.
    """
    weights = torch.max(weights, torch.zeros_like(weights))
    if torch.sum(weights) > 1e-8:
        return weights / torch.sum(weights)
    return torch.zeros_like(weights)


def softmax_normalize(weights, temperature=1.0):
    """
    It takes a vector of weights and returns a new vector of weights where each weight is the softmax of
    the original weight

    :param weights: the weights of the different actions
    :param temperature: The temperature parameter controls how much the softmax function "squashes" the
    outputs
    :return: The softmax function is being applied to the weights divided by the temperature.
    """
    return nn.functional.softmax(weights / temperature, dim=0)


# It's a class that keeps track of the average of a list of numbers
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# 10 classes
# run model, predict, actual class:1 predicted: class 1
# run model, [0.1, 0.2, 0.3, ....., 0.01]: actual class: 2, predicted: 3, Inaccurate
# top-2: actual class belongs to any of the top-2 probabilities. Accurate


def calc_f1(output, target):
    """
    It takes the output of the model and the target, and returns the F1 score

    :param output: the output of the model, which is a tensor of shape (batch_size, num_classes)
    :param target: the ground truth labels
    """
    with torch.no_grad():
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()

    return [metrics.f1_score(target.cpu(), pred.squeeze(0).cpu(), average="weighted")]


def calc_auroc(output, target):
    """
    It calculates the area under the ROC curve (AUROC) for a given model output and target

    :param output: the output of the model
    :param target: the ground truth labels
    :return: The area under the ROC curve.
    """
    with torch.no_grad():
        org_preds = output.softmax(dim=1)
        # org_preds = (org_preds[:,0] <0.5).int()

        # test_preds = org_preds.view(-1).cpu().detach().numpy()
        test_truth = target.view(-1).cpu().detach().numpy()

        fpr_roc, tpr_roc, thresholds_roc = metrics.roc_curve(
            test_truth, org_preds[:, 1].view(-1).cpu().detach().numpy(), pos_label=1
        )
        au_roc = metrics.auc(fpr_roc, tpr_roc)
        # print("AUC: {}".format(np.round(au_roc,4)))
        return [au_roc]

    #     precision, recall, _ = precision_recall_curve(test_truth, org_preds[:,1].view(-1).cpu().detach().numpy())
    #     au_pr = auc(recall, precision)
    #     print("AUPRC: {}".format(np.round(au_pr,4)))

    #     # ''auprc': au_pr,
    #     print("-" * 50)
    #     #'auc':au_roc,
    #     return {'f1':f1, 'acc':acc, 'auc':au_roc,'auprc': au_pr}

    #     # _, pred = output.topk(1, 1, True, True)
    #     # pred = pred.t()

    # try:
    #     au_roc = metrics.roc_auc_score(target.cpu(), pred.squeeze(0).cpu())
    # except:
    #     au_roc = 0.5

    # return [au_roc]


def calc_aupr(output, target):
    """
    It calculates the area under the precision-recall curve

    :param output: the output of the model, which is a tensor of shape (batch_size, num_classes)
    :param target: the ground truth labels
    :return: The area under the precision-recall curve.
    """
    with torch.no_grad():
        org_preds = output.softmax(dim=1)
        # org_preds = (org_preds[:,0] <0.5).int()

        # test_preds = org_preds.view(-1).cpu().detach().numpy()
        test_truth = target.view(-1).cpu().detach().numpy()

        precision, recall, _ = metrics.precision_recall_curve(
            test_truth, org_preds[:, 1].view(-1).cpu().detach().numpy()
        )
        au_pr = metrics.auc(recall, precision)
        # print("AUPRC: {}".format(np.round(au_pr,4)))
        return [au_pr]

    #     _, pred = output.topk(1, 1, True, True)
    #     pred = pred.t()

    # precision, recall, _ = metrics.precision_recall_curve(
    #     target.cpu(), pred.squeeze(0).cpu()
    # )

    # au_pr = metrics.auc(recall, precision)

    # return [au_pr]


def save_checkpoint(args, state, is_best):
    path = "checkpoint/ours/"
    save_name = args.save_name
    if not os.path.exists(path):
        os.makedirs(path)
    filename = path + save_name + "_ckpt.pth.tar"
    if is_best:
        torch.save(state, filename)


def to_categorical(labels):
    labels_onehot = torch.zeros([labels.shape[0], args.num_classes])
    for label_epoch in range(labels.shape[0]):
        labels_onehot[label_epoch, labels[label_epoch]] = 1

    return labels_onehot


if __name__ == "__main__":
    main()
