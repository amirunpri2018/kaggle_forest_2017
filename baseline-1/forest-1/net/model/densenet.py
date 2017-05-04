#https://github.com/hengck23-udacity/udacity-driverless-car-nd-p2/blob/master/advance/code/net/densenet.py
#https://github.com/andreasveit/densenet-pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.common import *
from net.utility.tool import *

class Block(nn.Module):
        def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)


class DenseNet(nn.Module):

    def make_conv_bn_relu(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),

            ##nn.ReLU(inplace=True),
            #nn.ELU(inplace=True),
        ]
