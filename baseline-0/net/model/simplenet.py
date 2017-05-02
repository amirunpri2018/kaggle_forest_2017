# https://www.kaggle.com/anokas/planet-understanding-the-amazon-from-space/simple-keras-starter
# 0.92 on leaderboard ??


import torch
import torch.nn as nn
import torch.nn.functional as F

from net.common import *
from net.utility.tool import *


class SimpleNet(nn.Module):

    def make_conv_bn_relu(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        layer  = []
        layer += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)]
        layer += [nn.BatchNorm2d(out_channels)]
        layer += [nn.ReLU(inplace=True)]
        return layer

    def __init__(self, in_channels, num_classes):
        super(VggNet, self).__init__()

        layer1 = []
        layer1 += self.make_conv_bn_relu(in_channels, 64)
        layer1 += self.make_conv_bn_relu(64, 64,)
        layer1 += [nn.MaxPool2d(kernel_size=2, stride=2)]
        self.layer1 = nn.Sequential(*layer1)

        layer2 = []
        layer2 += self.make_conv_bn_relu( 64, 128)
        layer2 += self.make_conv_bn_relu(128, 128)
        layer2 += [nn.MaxPool2d(kernel_size=2, stride=2)]
        self.layer2 = nn.Sequential(*layer2)

        layer3 = []
        layer3 += self.make_conv_bn_relu(128, 256)
        layer3 += self.make_conv_bn_relu(256, 256)
        layer3 += self.make_conv_bn_relu(256, 256)
        layer3 += [nn.MaxPool2d(kernel_size=2, stride=2)]
        self.layer3 = nn.Sequential(*layer3)

        layer4 = []
        layer4 += self.make_conv_bn_relu(256, 512)
        layer4 += self.make_conv_bn_relu(512, 512)
        layer4 += self.make_conv_bn_relu(512, 512)
        layer4 += [nn.MaxPool2d(kernel_size=2, stride=2)]
        self.layer4 = nn.Sequential(*layer4)

        layer5 = []
        layer5 += self.make_conv_bn_relu(512, 512)
        layer5 += self.make_conv_bn_relu(512, 512)
        layer5 += self.make_conv_bn_relu(512, 512)
        layer5 += [nn.MaxPool2d(kernel_size=2, stride=2)]
        self.layer5 = nn.Sequential(*layer5)

        self.layer6 = nn.AdaptiveAvgPool2d(1)  ## F.max_pool2d(input, kernel_size=input.size()[2:])

        ##dropout here
        self.logit  = nn.Linear(512, num_classes)


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = out.view(out.size(0), -1)
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
        x = Variable(inputs).cuda()

        start = timer()
        y = net.forward(x)
        end = timer()
        print ('cuda(): end-start=%0.0f  ms'%((end - start)*1000))

        dot = make_dot(y)
        dot.view()
        print(net)
        print(y)

