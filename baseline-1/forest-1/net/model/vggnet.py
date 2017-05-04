# https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.common import *
from net.utility.tool import *


class VggNet(nn.Module):

    # def make_conv_bn_relu(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    #     return [
    #         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
    #         nn.BatchNorm2d(out_channels),
    #         nn.ReLU(inplace=True),
    #     ]

    def make_conv_bn_prelu(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        ]

    def make_linear_bn_prelu(self, in_channels, out_channels):
        return [
            nn.Linear(in_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.PReLU(out_channels),
        ]

    def __init__(self, in_shape, num_classes):
        super(VggNet, self).__init__()
        in_channels, height, width = in_shape
        stride=1

        self.layer0 = nn.Sequential(
            *self.make_conv_bn_prelu(in_channels, 8, kernel_size=1, stride=1, padding=0 ),
            *self.make_conv_bn_prelu(8, 8, kernel_size=1, stride=1, padding=0 ),
            *self.make_conv_bn_prelu(8, 8, kernel_size=1, stride=1, padding=0 ),
        )

        self.layer1 = nn.Sequential(
            *self.make_conv_bn_prelu( 8, 64),
            *self.make_conv_bn_prelu(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.05),
        )
        stride*=2

        self.layer2 = nn.Sequential(
            *self.make_conv_bn_prelu( 64, 128),
            *self.make_conv_bn_prelu(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.05),
        )
        stride*=2

        self.layer3 = nn.Sequential(
            *self.make_conv_bn_prelu(128, 256),
            *self.make_conv_bn_prelu(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        stride*=2


        self.layer4 = nn.Sequential(
            *self.make_conv_bn_prelu(256, 512),
            *self.make_conv_bn_prelu(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),
            nn.AdaptiveAvgPool2d(1)
        )
        stride*=2

        self.layer5 = nn.Sequential(
            *self.make_linear_bn_prelu(512, 512),
            *self.make_linear_bn_prelu(512, 512),
            nn.Dropout(p=0.50),
        )
        self.logit = nn.Linear(512, num_classes)



    def forward(self, x):

        out  = self.layer0(x)
        out  = self.layer1(out)
        out  = self.layer2(out)
        out  = self.layer3(out)
        out  = self.layer4(out)

        out = out.view(out.size(0), -1)
        out = self.layer5(out)
        out = self.logit(out)

        return out


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    # https://discuss.pytorch.org/t/print-autograd-graph/692/8
    inputs = torch.randn(32,3,256,256)
    in_channels = inputs.size()[1]
    num_classes = 17


    if 1:
        net = VggNet(in_channels,num_classes).cuda()
        x   = Variable(inputs).cuda()

        start = timer()
        y  = net.forward(x)
        end = timer()
        print ('cuda(): end-start=%0.0f  ms'%((end - start)*1000))

        print(net)
        print(y)

