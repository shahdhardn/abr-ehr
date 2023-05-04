import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import torch.nn.init as init
import base
from typing import Any


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class resnet_attention(nn.Module):
    def __init__(self, enc_hid_dim=64, dec_hid_dim=100):
        super(resnet_attention, self).__init__()

        self.attn = nn.Linear(enc_hid_dim, dec_hid_dim, bias=True)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, s):
        energy = torch.tanh(self.attn(s))
        attention = self.v(energy)

        return F.softmax(attention, dim=0)


# check class meta module
class MetaModule(nn.Module):
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=""):
        if memo is None:
            memo = set()

        if hasattr(curr_module, "named_leaves"):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ("." if prefix else "") + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ("." if prefix else "") + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ("." if prefix else "") + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(
        self, lr_inner, first_order=False, source_params=None, detach=False
    ):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:
            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if "." in name:
            n = name.split(".")
            module_name = n[0]
            rest = ".".join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer("weight", to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer("bias", to_var(ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [("weight", self.weight), ("bias", self.bias)]


class MetaLinear_Norm(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        temp = nn.Linear(*args, **kwargs)
        temp.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.register_buffer("weight", to_var(temp.weight.data.t(), requires_grad=True))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

    def named_leaves(self):
        return [("weight", self.weight)]


class MetaConv2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Conv2d(*args, **kwargs)

        self.in_channels = ignore.in_channels
        self.out_channels = ignore.out_channels
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        self.kernel_size = ignore.kernel_size

        self.register_buffer("weight", to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer("bias", to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer("bias", None)

    def forward(self, x):
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def named_leaves(self):
        return [("weight", self.weight), ("bias", self.bias)]


class MetaConvTranspose2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.ConvTranspose2d(*args, **kwargs)

        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups

        self.register_buffer("weight", to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer("bias", to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer("bias", None)

    def forward(self, x, output_size=None):
        output_padding = self._output_padding(x, output_size)
        return F.conv_transpose2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )

    def named_leaves(self):
        return [("weight", self.weight), ("bias", self.bias)]


class MetaBatchNorm2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm2d(*args, **kwargs)

        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        if self.affine:
            self.register_buffer(
                "weight", to_var(ignore.weight.data, requires_grad=True)
            )
            self.register_buffer("bias", to_var(ignore.bias.data, requires_grad=True))

        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(self.num_features))
            self.register_buffer("running_var", torch.ones(self.num_features))
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_var", None)

    def forward(self, x):
        return F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training or not self.track_running_stats,
            self.momentum,
            self.eps,
        )

    def named_leaves(self):
        return [("weight", self.weight), ("bias", self.bias)]


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, MetaLinear) or isinstance(m, MetaConv2d):
        init.kaiming_normal(m.weight)
    elif isinstance(m, MetaLstm):
        init.kaiming_normal(m.weight_ih_l0)


class LambdaLayer(MetaModule):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(MetaModule):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option="A"):
        super(BasicBlock, self).__init__()
        self.conv1 = MetaConv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = MetaBatchNorm2d(planes)
        self.conv2 = MetaConv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = MetaBatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == "A":
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        "constant",
                        0,
                    )
                )
            elif option == "B":
                self.shortcut = nn.Sequential(
                    MetaConv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    MetaBatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet32(MetaModule):
    def __init__(self, num_classes, block=BasicBlock, num_blocks=[5, 5, 5]):
        super(ResNet32, self).__init__()
        self.in_planes = 16

        self.conv1 = MetaConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = MetaBatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = MetaLinear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return out, y


class BinaryClassification(MetaModule):
    def __init__(self, num_classes=2, num_features=96):
        super(BinaryClassification, self).__init__()
        self.layer_1 = nn.Linear(num_features, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.linear = MetaLinear(64, num_classes)

        self.apply(_weights_init)

        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(p=0.1)
        # self.batchnorm1 = nn.BatchNorm1d(64)
        # self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, inputs):
        inputs = inputs[0][0].to(torch.float32)
        x = self.relu(self.layer_1(inputs))
        # x = self.batchnorm1(x)
        x = self.layer_2(x)
        # x = self.batchnorm2(x)
        # x = self.dropout(x)

        return x, self.linear(x), torch.sigmoid(self.linear(x)).squeeze(1)


PRETRAIN_CHECKPOINT_PATH = "/l/users/mai.kassem/datasets/ClinicalBERT_pytorch_model.bin"


class MetaLstm(MetaModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.ignore = nn.LSTM(*args, **kwargs)
        self.input_size = self.ignore.input_size
        self.hidden_size = self.ignore.hidden_size
        self.num_layers = self.ignore.num_layers
        self.batch_first = True

        self.register_buffer(
            "weight_ih_l0", to_var(self.ignore.weight_ih_l0.data, requires_grad=True)
        )
        self.register_buffer(
            "weight_hh_l0", to_var(self.ignore.weight_hh_l0.data, requires_grad=True)
        )
        self.register_buffer(
            "bias_ih_l0", to_var(self.ignore.bias_ih_l0.data, requires_grad=True)
        )
        self.register_buffer(
            "bias_hh_l0", to_var(self.ignore.bias_hh_l0.data, requires_grad=True)
        )

        self.project = nn.Linear(self.hidden_size, 512)  # number of neurons is 512
        self.drop = nn.Dropout(0.0)  # dropout is 0.0

    def forward(self, x):
        # self.lstm.flatten_parameters()
        # h_all, (h_T, c_T) = self.lstm(x)
        dict_parms = {
            "weight_ih_l0": self.weight_ih_l0,
            "weight_hh_l0": self.weight_hh_l0,
            "bias_ih_l0": self.bias_ih_l0,
            "bias_hh_l0": self.bias_hh_l0,
            "num_layers": self.num_layers,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "batch_first": self.batch_first,
        }
        h_all, (h_T, c_T) = nn.utils.stateless.functional_call(
            self.ignore, dict_parms, (x)
        )
        output = h_T[-1]
        return F.relu(self.drop(self.project(output)))

    def named_leaves(self):
        return [("weight_ih_l0", self.weight_ih_l0), ("bias_ih_l0", self.bias_ih_l0)]


class MBertLstm(MetaModule):
    def __init__(
        self,
        pretrained_bert_dir: str = PRETRAIN_CHECKPOINT_PATH,
        ti_input_size: int = 96,
        ti_norm_size: int = 64,
        ts_input_size: int = 5132,
        ts_norm_size: int = 1024,
        n_neurons: int = 512,
        bert_size: int = 768,
        output_size: int = 1,
        num_layers: int = 1,
        dropout: int = 0.1,
        num_training_steps: int = 1000,
        warmup_proportion: float = 0.1,
        **kwargs: Any
    ) -> None:
        super().__init__()
        # self.save_hyperparameters()
        self.num_training_steps = num_training_steps
        self.warmup_proportion = warmup_proportion

        self.ti_enc = MetaLinear(ti_input_size, ti_norm_size)

        self.ts_enc = base.Lstm(
            input_size=ts_input_size,
            hidden_size=ts_norm_size,
            n_neurons=n_neurons,
            num_layers=num_layers,
        )
        # self.ts_enc = MetaLstm(
        #     input_size=ts_input_size, hidden_size=ts_norm_size, num_layers=num_layers
        # )

        self.nt_enc = base.Bert(pretrained_bert_dir=pretrained_bert_dir)

        self.gate = base.Gate(bert_size, ti_norm_size, n_neurons, dropout)

        # self.linear = nn.Linear(bert_size, output_size)
        self.linear = MetaLinear(bert_size, output_size)

        self.apply(_weights_init)

    def forward(self, x):
        ti = self.ti_enc(x[0][0])
        ts = self.ts_enc(x[0][1])
        nt = self.nt_enc(x[1:])
        fusion = self.gate(nt, ti, ts)
        # return features, logits, predictions
        return (
            fusion,
            self.linear(fusion),
            torch.sigmoid(self.linear(fusion)).squeeze(1),
        )
        # return (
        #     ti,
        #     self.linear(ti),
        #     torch.sigmoid(self.linear(ti)).squeeze(1),
        # )


class Line(MetaModule):
    def __init__(
        self,
        input_size: int = 96,
        hidden_size: int = 64,
        output_size: int = 2,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.enc = MetaLinear(input_size, hidden_size)
        self.linear = MetaLinear(hidden_size, output_size)
        self.apply(_weights_init)

    def forward(self, x):
        x = self.enc(x[0][0])
        return x, self.linear(x), torch.sigmoid(self.linear(x)).squeeze(1)
