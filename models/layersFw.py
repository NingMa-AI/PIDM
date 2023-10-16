import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class Linear_fw(nn.Linear):  # used in MAML to forward input with fast weight
    def __init__(self, in_features, out_features):
        super(Linear_fw, self).__init__(in_features, out_features)
        self.in_features=in_features
        self.out_features=out_features
        self.weight.fast = None  # Lazy hack to add fast weight link
        self.bias.fast = None

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            weights = Parameter(self.weight.data.clone(), self.weight.requires_grad)
            bias=Parameter(self.bias.data.clone(),self.bias.requires_grad)
            weights.fast=None
            bias.fast=None

            result=type(self)(self.in_features,self.out_features)
            result.weight=weights
            result.bias=bias

            memo[id(self)] = result
            return result

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast,
                           self.bias.fast)  # weight.fast (fast weight) is the temporaily adapted weight
        else:
            out = super(Linear_fw, self).forward(x)
        return out


class Conv2d_fw(nn.Conv2d):  # used in MAML to forward input with fast weight
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                        bias=bias)
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.curent_bias=bias

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            weights = Parameter(self.weight.data.clone(), self.weight.requires_grad)
            bias = Parameter(self.bias.data.clone(), self.bias.requires_grad)
            weights.fast = None
            bias.fast = None
            result = type(self)(self.in_channels, self.out_channels,self.kernel_size,self.stride,self.padding,self.curent_bias)
            result.weight = weights
            result.bias = bias
            memo[id(self)] = result
            return result

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)

        return out


class BatchNorm2d_fw(nn.BatchNorm2d):  # used in MAML to forward input with fast weight
    def __init__(self, num_features):
        super(BatchNorm2d_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None
    def forward(self, x):
        running_mean = torch.zeros(x.data.size()[1]).cuda()
        running_var = torch.ones(x.data.size()[1]).cuda()
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.batch_norm(x, running_mean, running_var, self.weight.fast, self.bias.fast, training=True,
                               momentum=1)
            # batch_norm momentum hack: follow hack of Kate Rakelly in pytorch-maml/src/layers.py
        else:
            out = F.batch_norm(x, running_mean, running_var, self.weight, self.bias, training=True, momentum=1)
        return out

class BatchNorm1d_fw(nn.BatchNorm1d):  # used in MAML to forward input with fast weight
    def __init__(self, num_features,affin=True):
        super(BatchNorm1d_fw, self).__init__(num_features,affine=affin)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        running_mean = torch.zeros(x.data.size()[1]).cuda()
        running_var = torch.ones(x.data.size()[1]).cuda()
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.batch_norm(x, running_mean, running_var, self.weight.fast, self.bias.fast, training=True,
                               momentum=1)
            # batch_norm momentum hack:follow hack of Kate Rakelly in pytor ch-maml/src/layers.py
        else:
            out = F.batch_norm(x, running_mean, running_var, self.weight, self.bias, training=True, momentum=1)
        return out
