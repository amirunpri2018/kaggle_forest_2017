# https://www.kaggle.com/anokas/planet-understanding-the-amazon-from-space/simple-keras-starter
# 0.92 on leaderboard ??

# https://discuss.pytorch.org/t/multi-label-classification-in-pytorch/905/13
# https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252
#


import torch
import torch.nn as nn
import torch.nn.functional as F

from net.common import *
from net.utility.tool import *


class SimpleNet(nn.Module):

    def make_conv_bn_relu(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

    def __init__(self, in_shape, num_classes):
        super(SimpleNet, self).__init__()
        in_channels, height, width = in_shape
        stride=1

        self.layer1 = nn.Sequential(
            *self.make_conv_bn_relu(in_channels,32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.25),
        )
        stride*=2

        self.layer2 = nn.Sequential(
            *self.make_conv_bn_relu(32,64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),
        )
        stride*=2

        self.layer3 = nn.Sequential(
            *self.make_conv_bn_relu(64,128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.50),
        )
        stride*=2

        self.layer5 = nn.Sequential(
            nn.Linear(128 * (height//stride) *(width//stride), 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

        self.logit = nn.Linear(512, num_classes)


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.layer5(out)
        out = self.logit(out)
        return out



class SimpleNet1(nn.Module):

    def make_conv_bn_relu(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

    def __init__(self, in_shape, num_classes):
        super(SimpleNet1, self).__init__()
        in_channels, height, width = in_shape
        stride=1

        self.layer1 = nn.Sequential(
            *self.make_conv_bn_relu(in_channels,32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.25),
        )
        stride*=2

        self.layer2 = nn.Sequential(
            *self.make_conv_bn_relu(32,64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),
        )
        stride*=2
        dim2 = 64 * (height//stride) *(width//stride)

        self.layer3 = nn.Sequential(
            *self.make_conv_bn_relu(64,128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.50),
        )
        stride*=2
        dim3 = 128 * (height//stride) *(width//stride)

        self.layer5 = nn.Sequential(
            nn.Linear(dim2+dim3, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

        self.logit = nn.Linear(512, num_classes)


    def forward(self, x):

        out = self.layer1(x)
        out = self.layer2(out)
        out2 = out.view(out.size(0), -1)

        out = self.layer3(out)
        out3 = out.view(out.size(0), -1)

        out = torch.cat([out2,out3],1)
        out = self.layer5(out)
        out = self.logit(out)

        return out




class SimpleNet2(nn.Module):

    def make_conv_bn_relu(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
    def make_linear_bn_relu(self, in_channels, out_channels):
        return [
            nn.Linear(in_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        ]
    def make_conv_bn_prelu(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        ]


    def __init__(self, in_shape, num_classes):
        super(SimpleNet2, self).__init__()
        in_channels, height, width = in_shape
        stride=1

        self.layer0 = nn.Sequential(
            *self.make_conv_bn_prelu(in_channels, 8, kernel_size=1, stride=1, padding=0 ),
            *self.make_conv_bn_prelu(8, 8, kernel_size=1, stride=1, padding=0 ),
            *self.make_conv_bn_prelu(8, 8, kernel_size=1, stride=1, padding=0 ),
        )

        self.layer1 = nn.Sequential(
            *self.make_conv_bn_prelu( 8, 32),
            *self.make_conv_bn_prelu(32, 32, kernel_size=1, stride=1, padding=0 ),
            *self.make_conv_bn_prelu(32, 32, kernel_size=1, stride=1, padding=0 ),
            *self.make_conv_bn_prelu(32, 32, kernel_size=1, stride=1, padding=0 ),
            *self.make_conv_bn_prelu(32, 32, kernel_size=3, stride=1, padding=1 ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.25),
        )
        stride*=2

        self.layer2 = nn.Sequential(
            *self.make_conv_bn_prelu(32, 64),
            *self.make_conv_bn_prelu(64, 64, kernel_size=1, stride=1, padding=0 ),
            *self.make_conv_bn_prelu(64, 64, kernel_size=1, stride=1, padding=0 ),
            *self.make_conv_bn_prelu(64, 64, kernel_size=1, stride=1, padding=0 ),
            *self.make_conv_bn_prelu(64, 64, kernel_size=3, stride=1, padding=1 ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),
        )
        stride*=2
        dim2 = 64 * (height//stride) *(width//stride)

        self.layer3 = nn.Sequential(
            *self.make_conv_bn_prelu( 64, 128),
            *self.make_conv_bn_prelu(128, 128, kernel_size=1, stride=1, padding=0 ),
            *self.make_conv_bn_prelu(128, 128, kernel_size=1, stride=1, padding=0 ),
            *self.make_conv_bn_prelu(128, 128, kernel_size=1, stride=1, padding=0 ),
            *self.make_conv_bn_prelu(128, 128, kernel_size=3, stride=1, padding=1 ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.50),
        )
        stride*=2
        dim3 = 128 * (height//stride) *(width//stride)

        self.layer5 = nn.Sequential(
            *self.make_linear_bn_relu(64+128, 512),
            *self.make_linear_bn_relu(512, 512)
        )

        self.logit = nn.Linear(512, num_classes)


    def forward(self, x):

        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)

        out2 = nn.AdaptiveAvgPool2d(1)(out)
        out2 = out2.view(out2.size(0), -1)

        out = self.layer3(out)

        out3 = nn.AdaptiveAvgPool2d(1)(out)
        out3 = out3.view(out3.size(0), -1)

        out = torch.cat([out2,out3],1)
        out = self.layer5(out)
        out = self.logit(out)

        return out




class SimpleNet64_2(nn.Module):

    def make_conv_bn_relu(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
    def make_linear_bn_relu(self, in_channels, out_channels):
        return [
            nn.Linear(in_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        ]
    def make_conv_bn_prelu(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        ]


    def __init__(self, in_shape, num_classes):
        super(SimpleNet64_2, self).__init__()
        in_channels, height, width = in_shape
        stride=1

        self.layer0 = nn.Sequential(
            *self.make_conv_bn_prelu(in_channels, 8, kernel_size=1, stride=1, padding=0 ),
            *self.make_conv_bn_prelu(8, 8, kernel_size=1, stride=1, padding=0 ),
            *self.make_conv_bn_prelu(8, 8, kernel_size=1, stride=1, padding=0 ),
        )

        self.layer1 = nn.Sequential(
            *self.make_conv_bn_prelu( 8, 32),
            *self.make_conv_bn_prelu(32, 32, kernel_size=1, stride=1, padding=0 ),
            *self.make_conv_bn_prelu(32, 32, kernel_size=1, stride=1, padding=0 ),
            *self.make_conv_bn_prelu(32, 32, kernel_size=1, stride=1, padding=0 ),
            *self.make_conv_bn_prelu(32, 32, kernel_size=3, stride=1, padding=1 ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.25),
        )
        stride*=2

        self.layer2 = nn.Sequential(
            *self.make_conv_bn_prelu(32, 32),
            *self.make_conv_bn_prelu(32, 32, kernel_size=1, stride=1, padding=0 ),
            *self.make_conv_bn_prelu(32, 32, kernel_size=1, stride=1, padding=0 ),
            *self.make_conv_bn_prelu(32, 32, kernel_size=1, stride=1, padding=0 ),
            *self.make_conv_bn_prelu(32, 32, kernel_size=3, stride=1, padding=1 ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.25),
        )
        stride*=2

        self.layer3 = nn.Sequential(
            *self.make_conv_bn_prelu(32, 64),
            *self.make_conv_bn_prelu(64, 64, kernel_size=1, stride=1, padding=0 ),
            *self.make_conv_bn_prelu(64, 64, kernel_size=1, stride=1, padding=0 ),
            *self.make_conv_bn_prelu(64, 64, kernel_size=1, stride=1, padding=0 ),
            *self.make_conv_bn_prelu(64, 64, kernel_size=3, stride=1, padding=1 ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),
        )
        stride*=2
        dim2 = 64 * (height//stride) *(width//stride)

        self.layer4 = nn.Sequential(
            *self.make_conv_bn_prelu( 64, 128),
            *self.make_conv_bn_prelu(128, 128, kernel_size=1, stride=1, padding=0 ),
            *self.make_conv_bn_prelu(128, 128, kernel_size=1, stride=1, padding=0 ),
            *self.make_conv_bn_prelu(128, 128, kernel_size=1, stride=1, padding=0 ),
            *self.make_conv_bn_prelu(128, 128, kernel_size=3, stride=1, padding=1 ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.50),
        )
        stride*=2
        dim3 = 128 * (height//stride) *(width//stride)



        self.layer5 = nn.Sequential(
            *self.make_linear_bn_relu(64+128, 512),
            *self.make_linear_bn_relu(512, 512)
        )

        self.logit = nn.Linear(512, num_classes)


    def forward(self, x):

        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        outa = nn.AdaptiveAvgPool2d(1)(out)
        outa = outa.view(outa.size(0), -1)

        out = self.layer4(out)

        outb = nn.AdaptiveAvgPool2d(1)(out)
        outb = outb.view(outb.size(0), -1)

        out = torch.cat([outa,outb],1)
        out = self.layer5(out)
        out = self.logit(out)

        return out


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    # https://discuss.pytorch.org/t/print-autograd-graph/692/8
    inputs = torch.randn(32,3,32,32)
    in_shape = inputs.size()[1:]
    num_classes = 17

    if 1:
        net = SimpleNet2(in_shape,num_classes).cuda()
        x = Variable(inputs).cuda()

        start = timer()
        y = net.forward(x)
        end = timer()
        print ('cuda(): end-start=%0.0f  ms'%((end - start)*1000))

        #dot = make_dot(y)
        #dot.view()
        print(net)
        print(y)

