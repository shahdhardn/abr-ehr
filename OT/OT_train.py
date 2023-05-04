import os
import time
import argparse
import random
import math
import copy
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
import torchvision
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import shutil
from Sinkhorn_distance import SinkhornDistance
from Sinkhorn_distance_fl import SinkhornDistance as SinkhornDistance_fl
from sklearn import metrics
from data_module import *

from models import *

from data_module import MimicDataModule
from data_utils import *

# Log in to your W&B account
import wandb

wandb.login()

parser = argparse.ArgumentParser(description="Imbalanced Example")
parser.add_argument(
    "--dataset",
    default="mimic",
    type=str,
    help="mimic",
)
parser.add_argument(
    "--cost", default="combined", type=str, help="[combined, label, feature, twoloss]"
)
parser.add_argument("--meta_set", default="whole", type=str, help="[whole, prototype]")
parser.add_argument(
    "--batch_size",
    type=int,
    default=40,
    metavar="N",
    help="input batch size for training (default: 16)",
)
parser.add_argument("--num_classes", type=int, default=2)
parser.add_argument(
    "--num_meta", type=int, default=10, help="The number of meta data for each class."
)
parser.add_argument("--imb_factor", type=float, default=0.08)
parser.add_argument(
    "--epochs", type=int, default=20, metavar="N", help="number of epochs to train"
)
parser.add_argument(
    "--start_epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
# parser.add_argument(
#     "--lr", "--learning_rate", default=1e-4, type=float, help="initial learning rate"
# )
parser.add_argument(
    "--lr", "--learning_rate", default=0.1, type=float, help="initial learning rate"
)
parser.add_argument(
    "--cos", default=False, type=bool, help="lr decays by cosine scheduler. "
)
# TODO change number according to the number of epochs
parser.add_argument(
    "--schedule",
    default=[15, 18],
    nargs="*",
    type=int,
    help="learning rate schedule (when to drop lr by 10x)",
)
parser.add_argument("--warmup_epochs", default=0, type=int, help="warmup epochs")
parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
parser.add_argument("--nesterov", default=True, type=bool, help="nesterov momentum")
parser.add_argument(
    "--weight_decay",
    "--wd",
    default=5e-4,
    type=float,
    help="weight decay (default: 5e-4)",
)
parser.add_argument(
    "--no_cuda", action="store_true", default=False, help="disables CUDA training"
)
parser.add_argument(
    "--print_freq", "-p", default=100, type=int, help="print frequency (default: 100)"
)
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument("--save_name", default="OT_ARF12", type=str)
parser.add_argument("--idx", default="ours", type=str)

parser.add_argument("--model", type=str, default="MBertLstm")
parser.add_argument(
    "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)

args = parser.parse_args()
for arg in vars(args):
    print("{}={}".format(arg, getattr(args, arg)))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
kwargs = {"num_workers": 16, "pin_memory": False}
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
pos_weights = torch.tensor(
    [1 - (x / sum(class_counts)) for x in class_counts], device=device
)
weightsbuffer = torch.tensor(
    [per_cls_weights[cls_i] for cls_i in imbalanced_train_labels]
).to(device)
del per_cls_weights

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
args.save_name = f"exp_{run}_{args.model}_{args.batch_size}"
wandb.init(
    entity="abr-ehr",
    # Set the project where this run will be logged
    project="OT_ARF12",
    # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    name=args.save_name,
    # Track hyperparameters and run metadata
    config={
        "learning_rate": args.lr,
        "architecture": args.model,
        "dataset": args.dataset,
        "epochs": args.epochs,
        "imb_factor": args.imb_factor,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
    },
)


def main():
    global args, best_prec1, best_auroc, best_aupr, best_f1
    args = parser.parse_args()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))

            model = build_model(load_pretrain=True, ckpt_path=args.resume)
        else:
            model = build_model(load_pretrain=False, ckpt_path=None)
    else:
        model = build_model(load_pretrain=False, ckpt_path=None)

    # TODO optimizer

    # optimizer_a = torch.optim.SGD(
    #     [model.linear.weight, model.linear.bias],
    #     lr=args.lr,
    #     momentum=args.momentum,
    #     nesterov=args.nesterov,
    #     weight_decay=args.weight_decay,
    # )
    # optimizer_a = torch.optim.Adam(
    #     [model.linear.weight, model.linear.bias],
    #     lr=args.lr,
    #     weight_decay=args.weight_decay,
    # )

    # optimizer_a = torch.optim.AdamW(
    #     [model.linear.weight, model.linear.bias], lr=args.lr
    # )
    optimizer_a = torch.optim.AdamW(model.parameters(), lr=args.lr)

    cudnn.benchmark = True
    # criterion_classifier = nn.CrossEntropyLoss(reduction="none").to(device)
    criterion_classifier = nn.BCEWithLogitsLoss(reduction="none").to(device)

    for epoch in range(args.start_epoch, args.epochs):
        train_OT(
            imbalanced_train_loader,
            meta_loader,
            weightsbuffer,
            model,
            optimizer_a,
            epoch,
            criterion_classifier,
        )

        # if epoch > 1:
        #     test_model = combine_models(model)
        #     prec1, f1, auroc, aupr, preds, gt_labels = validate(test_loader, test_model)
        # else:
        prec1, f1, auroc, aupr, preds, gt_labels = validate(test_loader, model)

        is_best = auroc > best_auroc
        best_auroc = max(auroc, best_auroc)

        if is_best:
            weightsbuffer_bycls = []
            for i_cls in range(args.num_classes):
                weightsbuffer_bycls.extend(
                    weightsbuffer[imbalanced_train_labels == i_cls].data.cpu().numpy()
                )
            corresponding_prec1 = prec1
            corresponding_f1 = f1
            corresponding_aupr = aupr

        save_checkpoint(
            args,
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_auroc": best_auroc,
                "corresponding_prec1": corresponding_prec1,
                "corresponding_f1": corresponding_f1,
                "corresponding_aupr": corresponding_aupr,
                "optimizer": optimizer_a.state_dict(),
                "weights": weightsbuffer_bycls,
            },
            is_best,
        )

    print(
        "Best AUROC: ",
        best_auroc,
        "Corresponding accuracy: ",
        corresponding_prec1,
        "Corresponding F1: ",
        corresponding_f1,
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

    if args.cos:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10, eta_min=1e-6
        )
        # max_epoch = lr_lambda(epoch)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer, lr_lambda=lambda epoch: max_epoch
        # )
    else:
        adjust_lr(optimizer, epoch)
        print("Overall lr: ", optimizer.param_groups[0]["lr"])

    model.train()

    val_data, val_labels, _ = next(iter(validation_loader))
    print(np.unique(val_labels.cpu(), return_counts=True))
    val_data[0][0], val_data[0][1], val_data[1], val_data[2], val_data[3] = to_var_x(
        val_data, requires_grad=False
    )
    val_labels = to_var(val_labels, requires_grad=False).squeeze()

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
    feature_val, _, _ = model(val_data_bycls)
    del val_data_bycls
    for i, batch in enumerate(train_loader):
        inputs, labels, ids = tuple(t for t in batch)
        inputs[0][0], inputs[0][1], inputs[1], inputs[2], inputs[3] = to_var_x(
            inputs, requires_grad=False
        )

        labels = labels.to(device)
        ids = ids.to(device)
        labels = labels.squeeze()
        labels_onehot = to_categorical(labels).to(device)

        weights = to_var(weightsbuffer[ids])
        # model.eval()
        # Attoptimizer = torch.optim.SGD(
        #     [weights],
        #     lr=args.lr,
        #     momentum=args.momentum,
        #     weight_decay=args.weight_decay,
        # )
        Attoptimizer = torch.optim.Adam(
            [weights], lr=args.lr, weight_decay=args.weight_decay
        )
        # if args.cos:
        ot_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            Attoptimizer, T_max=10, eta_min=1e-6
        )
        for ot_epoch in range(10):
            # if not args.cos:
            # adjust_lr(Attoptimizer, ot_epoch)

            feature_train, _, _ = model(inputs)
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
            # if args.cos:
            ot_scheduler.step()

        weightsbuffer[ids] = weights.data

        model.train()
        optimizer.zero_grad()
        _, logits, probs = model(inputs)
        loss_train = criterion_classifier(logits.squeeze(1), labels.float())
        _, logits_val, probs_val = model(val_data)
        loss_val = F.binary_cross_entropy_with_logits(
            logits_val.squeeze(1), val_labels.float(), reduction="none"
        )
        # TODO: alpha scale to otloss
        # loss = torch.sum(loss_train * weightsbuffer[ids]) + 10 * torch.mean(loss_val)
        # loss = torch.sum(loss_train)
        loss = torch.mean(loss_train) + OTloss.item()
        loss.backward(retain_graph=False)
        optimizer.step()
        if args.cos:
            scheduler.step()
            # print current lr
            print("Overall lr: ", scheduler.get_lr())

        del weights

        # prec_train = accuracy(logits.data, labels, topk=(1,))[0]
        prec_train = calc_acc(probs, labels)[0]
        f1_acc = calc_f1(probs, labels)[0]
        try:
            au_roc = calc_auroc(probs, labels)[0]
        except Exception as ex:
            print("AUROC_ERROR in batch number: ", i, ex.__class__.__name__)
            au_roc = [0.5]
        au_pr = calc_aupr(probs, labels)[0]

        otlosses.update(OTloss.item(), labels.size(0))
        losses.update(loss.item(), labels.size(0))
        train_losses.update(loss_train.mean().item(), labels.size(0))
        val_losses.update(loss_val.mean().item(), labels.size(0))
        top1.update(prec_train.item(), labels.size(0))
        f1.update(f1_acc.item(), labels.size(0))
        auroc.update(au_roc, labels.size(0))
        aupr.update(au_pr.item(), labels.size(0))
        del prec_train, f1_acc, au_roc, au_pr

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
            "overall-loss": losses.avg,
            "CE-loss": train_losses.avg,
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
        # input[0][0] = torch.autograd.Variable(input[0][0])
        # input[0][1] = torch.autograd.Variable(input[0][1])
        # input[1] = torch.autograd.Variable(input[1])
        # input[2] = torch.autograd.Variable(input[2])
        # input[3] = torch.autograd.Variable(input[3])

        # target = torch.autograd.Variable(target)

        with torch.no_grad():
            _, output, probs = model(input)

        output_numpy = output.data.cpu().numpy()
        preds_output = list(output_numpy.argmax(axis=1))

        true_labels += list(target.data.cpu().numpy())
        preds += preds_output

        targets_var = to_var(target, requires_grad=False).squeeze()
        targets_var_onehot = to_categorical(targets_var).to(device)
        loss = torch.tensor(
            F.binary_cross_entropy_with_logits(
                output.data.squeeze(1), targets_var.float(), reduction="none"
            )
        )
        # prec1 = accuracy(output.data, target, topk=(1,))[0]
        prec1 = calc_acc(probs, target)[0]
        f1_acc = calc_f1(probs, target)[0]
        try:
            au_roc = calc_auroc(probs, target)[0]
        except Exception as ex:
            print("AUROC_ERROR in batch number: ", i, ex.__class__.__name__)
            au_roc = [0.5]
        au_pr = calc_aupr(probs, target)[0]

        losses.update(torch.mean(loss), target.size(0))
        top1.update(prec1.item(), target.size(0))
        f1.update(f1_acc.item(), target.size(0))
        auroc.update(au_roc, target.size(0))
        aupr.update(au_pr.item(), target.size(0))

        del prec1, f1_acc, au_roc, au_pr

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


def build_model(load_pretrain, ckpt_path=None, optimizer_a=None):
    """
    :param load_pretrain: whether to load pretrained model or not
    :param ckpt_path: the path to the checkpoint file
    :return: The model is being returned.
    """

    if args.model == "MBertLstm":
        if use_cuda:
            model = _CustomDataParallel(MBertLstm()).to(device)
            torch.backends.cudnn.benchmark = True
        else:
            model = MBertLstm()
    if args.model == "Line":
        if use_cuda:
            model = _CustomDataParallel(Line()).to(device)
            torch.backends.cudnn.benchmark = True
        else:
            model = Line()
    if args.model == "BinaryClassification":
        if use_cuda:
            model = _CustomDataParallel(BinaryClassification()).to(device)
            torch.backends.cudnn.benchmark = True
        else:
            model = BinaryClassification()
    if load_pretrain == True:
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        # optimizer_a.load_state_dict(checkpoint["optimizer"])
        args.start_epoch = checkpoint["epoch"]
        best_auroc = checkpoint["best_auroc"]

    return model


def combine_models(current_model):
    path = "checkpoint/ours/"
    save_name = args.save_name
    best_ckpt_path = path + save_name + "_best_ckpt.pth.tar"
    last_ckpt_path = path + save_name + "_last_ckpt.pth.tar"
    best_model_state_dict = torch.load(best_ckpt_path)["state_dict"]
    last_model_state_dict = torch.load(last_ckpt_path)["state_dict"]

    combined_state_dict = {
        k: (best_model_state_dict[k] + last_model_state_dict[k]) / 2
        for k in best_model_state_dict.keys() & last_model_state_dict.keys()
    }
    # Load the combined state dict
    current_model.load_state_dict(combined_state_dict)

    return current_model


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

    # if requires_grad:
    #     return (
    #         torch.autograd.Variable(ti),
    #         torch.autograd.Variable(ts),
    #         torch.autograd.Variable(bert_sent),
    #         torch.autograd.Variable(bert_sent_type),
    #         torch.autograd.Variable(bert_sent_mask),
    #     )
    if requires_grad:
        return (
            Variable(ti, requires_grad=requires_grad),
            Variable(ts, requires_grad=requires_grad),
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


def calc_f1(prob, target):
    """
    It takes the output of the model and the target, and returns the F1 score

    :param output: the output of the model, which is a tensor of shape (batch_size, num_classes)
    :param target: the ground truth labels
    """
    # with torch.no_grad():
    # prob = F.sigmoid(output)
    # pred = (prob[:, 0] > 0.5).int()
    pred = (prob > 0.5).int()
    pred = pred.view(-1).cpu().detach().numpy()
    target = target.view(-1).cpu().detach().numpy()

    return [metrics.f1_score(target, pred, average="macro")]


def calc_auroc(probabilities, target):
    """
    It calculates the area under the ROC curve (AUROC) for a given model output and target

    :param output: the output of the model
    :param target: the ground truth labels
    :return: The area under the ROC curve.
    """

    # with torch.no_grad():
    # probabilities = F.softmax(output, dim=1)[:, 1]
    # probabilities = F.sigmoid(output)[:, 1]
    # probabilities = probabilities[:, 1].cpu().detach().numpy()
    probabilities = probabilities.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    try:
        au_roc = metrics.roc_auc_score(target, probabilities, average="macro")
    except:
        au_roc = 0.5

    return [au_roc]


def calc_aupr(probabilities, target):
    """
    It calculates the area under the precision-recall curve

    :param output: the output of the model, which is a tensor of shape (batch_size, num_classes)
    :param target: the ground truth labels
    :return: The area under the precision-recall curve.
    """
    # with torch.no_grad():
    # probabilities = F.softmax(output, dim=1)[:, 1]
    # probabilities = F.sigmoid(output)[:, 1]
    # probabilities = probabilities[:, 1].cpu().detach().numpy()
    probabilities = probabilities.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    precision, recall, _ = metrics.precision_recall_curve(target, probabilities)

    au_pr = metrics.auc(recall, precision)

    return [au_pr]


def calc_acc(prob, target):
    # prob = F.sigmoid(output)
    # pred = (prob[:, 0] > 0.5).int()
    pred = (prob > 0.5).int()
    pred = pred.view(-1).cpu().detach().numpy()
    target = target.view(-1).cpu().detach().numpy()
    # print("max prob:", prob.max())
    # print("min prob:", prob.min())

    return [metrics.accuracy_score(target, pred)]


def save_checkpoint(args, state, is_best):
    path = "checkpoint/ours/"
    save_name = args.save_name
    if not os.path.exists(path):
        os.makedirs(path)
    if is_best:
        filename = path + save_name + "_best_ckpt.pth.tar"
    else:
        filename = path + save_name + "_last_ckpt.pth.tar"
    torch.save(state, filename)


def to_categorical(labels):
    labels_onehot = torch.zeros([labels.shape[0], args.num_classes])
    for label_epoch in range(labels.shape[0]):
        labels_onehot[label_epoch, labels[label_epoch]] = 1

    return labels_onehot


def adjust_lr(optimizer, epoch):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if epoch < args.warmup_epochs:
        lr = lr / args.warmup_epochs * (epoch + 1)
    elif args.cos:  # cosine lr schedule
        lr *= 0.5 * (
            1.0
            + math.cos(
                math.pi
                * (epoch - args.warmup_epochs + 1)
                / (args.epochs - args.warmup_epochs + 1)
            )
        )
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def lr_lambda(current_epoch):
    current_step = (
        current_epoch * len(imbalanced_train_loader.dataset.y) // args.batch_size
    )
    # num_training_steps = (
    #     args.epochs * len(imbalanced_train_loader.dataset.y) // args.batch_size
    # )
    num_training_steps = 1000
    num_warmup_steps = 0.1
    # num_warmup_steps = int(num_training_steps * 0.1)
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(
        0.0,
        float(num_training_steps - current_step)
        / float(max(1, num_training_steps - num_warmup_steps)),
    )


if __name__ == "__main__":
    main()
