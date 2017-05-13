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


## maxout ##
'''
https://github.com/jzbontar/jzt-old/blob/master/SpatialMaxout.lua
https://www.reddit.com/r/MachineLearning/comments/4o2njp/how_to_implement_maxout_in_torch/
https://github.com/pytorch/pytorch/issues/805
'''

class Maxout1d(nn.Module):

    def __init__(self, in_channels, out_channels, pool_size):
        super(Maxout1d, self).__init__()
        self.in_channels, self.out_channels, self.pool_size = in_channels, out_channels, pool_size
        self.linear = nn.Linear(in_channels, out_channels * pool_size)
        self.bn = nn.BatchNorm1d(out_channels * pool_size)

    def forward(self, x):
        N,C = list(x.size())
        out = self.linear(x)
        out = self.bn(out)
        m, i = out.view(N, self.out_channels, self.pool_size).max(2)
        m=m.squeeze(2)
        return m


class Maxout2d(nn.Module):

    def __init__(self, in_channels, out_channels, pool_size):
        super(Maxout2d, self).__init__()
        self.in_channels, self.out_channels, self.pool_size = in_channels, out_channels, pool_size
        self.conv2d = nn.Conv2d(in_channels, out_channels * pool_size, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(out_channels * pool_size)

    def forward(self, x):
        N,C,H,W  = list(x.size())
        out = self.conv2d(x)
        out = self.bn(out)

        out = out.permute(0, 2, 3, 1)
        m, i = out.contiguous().view(N*H*W,self.out_channels,self.pool_size).max(2)
        m = m.squeeze(2)
        m = m.view(N,H,W,self.out_channels)
        m = m.permute(0, 3, 1, 2)
        return m

##
def run_test_Maxout1d():
    inputs = torch.randn(32,5)
    in_channels  = 5
    out_channels = 3
    pool_size    = 4

    layer = Maxout1d(in_channels, out_channels, pool_size)
    outputs = layer(Variable(inputs))

    print(layer)
    print(outputs)
    print('in_channels=%d, out_channels=%d, pool_size=%d'%(in_channels,out_channels,pool_size))


def run_test_Maxout2d():
    inputs = torch.randn(32,5,10,10)
    in_channels  = 5
    out_channels = 3
    pool_size    = 4

    layer = Maxout2d(in_channels, out_channels, pool_size)
    outputs = layer(Variable(inputs))

    print(layer)
    print(outputs)
    print('in_channels=%d, out_channels=%d, pool_size=%d'%(in_channels,out_channels,pool_size))



###############################################################################################
def make_bn_conv_maxout(in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool_size=2):
    return [
        nn.BatchNorm2d(in_channels),
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        Maxout2d (out_channels, out_channels, pool_size),
        nn.ReLU(inplace=True),
    ]


###############################################################################################
def make_conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]


def make_linear_bn_relu(in_channels, out_channels):
    return [
        nn.Linear(in_channels, out_channels, bias=False),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
    ]

def make_conv_bn_prelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.PReLU(out_channels),
    ]

def make_linear_bn_prelu(in_channels, out_channels):
    return [
        nn.Linear(in_channels, out_channels, bias=False),
        nn.BatchNorm1d(out_channels),
        nn.PReLU(out_channels),
    ]


def make_conv_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        nn.ReLU(inplace=True),
    ]







def make_flat(out):
    flat = nn.AdaptiveMaxPool2d(1)(out)  #AdaptiveAvePool2d
    flat = flat.view(flat.size(0), -1)
    return flat

class SimpleNet_2cls_0(nn.Module):

    def __init__(self, in_shape, num_classes):
        super(SimpleNet_2cls_1, self).__init__()
        in_channels, height, width = in_shape
        stride=1

        self.block0 = nn.Sequential(
            *make_conv_bn_prelu(in_channels, 8, kernel_size=1, stride=1, padding=0 ),   ##  make_bn_conv_maxout
            *make_conv_bn_prelu(8, 8, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_prelu(8, 8, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_prelu(8, 8, kernel_size=1, stride=1, padding=0 ),
        )

        self.block1 = nn.Sequential(
            *make_conv_bn_prelu( 8, 32),
            *make_conv_bn_prelu(32, 32, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_prelu(32, 32, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_prelu(32, 32, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_prelu(32, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        stride*=2

        self.block2 = nn.Sequential(
            *make_conv_bn_prelu(32, 32),
            *make_conv_bn_prelu(32, 32, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_prelu(32, 32, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_prelu(32, 32, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_prelu(32, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.08),
        )
        stride*=2

        self.block3 = nn.Sequential(
            *make_conv_bn_prelu(32, 32),
            *make_conv_bn_prelu(32, 32, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_prelu(32, 32, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_prelu(32, 32, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_prelu(32, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.08),
        )
        stride*=2

        self.block4 = nn.Sequential(
            *make_conv_bn_prelu(32, 64),
            *make_conv_bn_prelu(64, 64, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_prelu(64, 64, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_prelu(64, 64, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_prelu(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.08),
        )
        stride*=2

        self.block5 = nn.Sequential(
            *make_conv_bn_prelu(64, 64),
            *make_conv_bn_prelu(64, 64, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_prelu(64, 64, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_prelu(64, 64, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_prelu(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),   ##F.max_pool2d  ##F.dropout(x, training=self.training)
            nn.Dropout(p=0.08),
        )
        stride*=2

        self.block6 = nn.Sequential(
            *make_conv_bn_prelu( 64, 128),
            *make_conv_bn_prelu(128, 128, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_prelu(128, 128, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_prelu(128, 128, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_prelu(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.08),
        )
        stride*=2

        self.block7 = nn.Sequential(
            *make_conv_bn_prelu(128, 128),
            *make_conv_bn_prelu(128, 128, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_prelu(128, 128, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_prelu(128, 128, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_prelu(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.08),
        )
        stride*=2

        self.block8 = nn.Sequential(
            *make_linear_bn_prelu(32+32+64+128+128, 512),
            *make_linear_bn_prelu(512, 512),
        )

        self.prob = nn.Sequential(
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )


    def forward(self, x):

        out = self.block0(x)
        out = self.block1(out)
        out = self.block2(out)
        flat2 = make_flat(out)

        out = self.block3(out)
        flat3 = make_flat(out)

        out = self.block4(out)
        flat4 = make_flat(out)

        # out = self.block5(out)
        # flat5 = make_flat(out)

        out = self.block6(out)
        flat6 = make_flat(out)

        out = self.block7(out)
        flat7 = make_flat(out)

        out = torch.cat([flat2, flat3, flat4, flat6, flat7],1)
        out = self.block8(out)
        out = self.prob(out)

        return out


class SimpleNet_2cls_2(nn.Module):

    def __init__(self, in_shape, num_classes):
        super(SimpleNet_2cls_2, self).__init__()
        in_channels, height, width = in_shape
        stride=1

        self.block0 = nn.Sequential(
            *make_conv_bn_prelu(in_channels, 8, kernel_size=1, stride=1, padding=0 ),   ##  make_bn_conv_maxout
        )

        self.block1 = nn.Sequential(
            *make_conv_bn_relu( 8, 32, stride=2),
        )
        stride*=2

        self.block2 = nn.Sequential(
            *make_conv_bn_relu(32, 32, stride=2),
        )
        stride*=2


        self.block3 = nn.Sequential(
            *make_conv_bn_relu(32, 64, stride=2),
        )
        stride*=2


        self.block8 = nn.Sequential(
            *make_linear_bn_relu(64, 512),
        )

        self.prob = nn.Sequential(
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )


    def forward(self, x):

        out = self.block0(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        flat3 = make_flat(out)
        out = self.block8(flat3)
        out = self.prob(out)

        return out


class SimpleNet_2cls_1(nn.Module):

    def __init__(self, in_shape, num_classes):
        super(SimpleNet_2cls_1, self).__init__()
        in_channels, height, width = in_shape
        stride=1

        self.block0 = nn.Sequential(
            *make_conv_bn_relu(in_channels, 8, kernel_size=1, stride=1, padding=0 ),   ##  make_bn_conv_maxout
        )

        self.block1 = nn.Sequential(
            *make_conv_bn_prelu( 8, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        stride*=2

        self.block2 = nn.Sequential(
            *make_conv_bn_prelu(32, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ##nn.Dropout(p=0.08),
        )
        stride*=2


        self.block3 = nn.Sequential(
            *make_conv_bn_prelu(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),   ##F.max_pool2d  ##F.dropout(x, training=self.training)
            #nn.Dropout(p=0.08),
        )
        stride*=2


        self.block8 = nn.Sequential(
            *make_linear_bn_prelu(64, 512),
        )

        self.prob = nn.Sequential(
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )


    def forward(self, x):

        out = self.block0(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        flat3 = make_flat(out)
        out = self.block8(flat3)
        out = self.prob(out)

        return out


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    if 0:
        run_test_Maxout1d()
        exit(0)

    # https://discuss.pytorch.org/t/print-autograd-graph/692/8
    inputs = torch.randn(32,3,72,72)
    in_shape = inputs.size()[1:]
    num_classes = 1

    if 1:
        net = SimpleNet_2cls_1(in_shape,num_classes).cuda()
        x = Variable(inputs).cuda()

        start = timer()
        y = net.forward(x)
        end = timer()
        print ('cuda(): end-start=%0.0f  ms'%((end - start)*1000))

        #dot = make_dot(y)
        #dot.view()
        print(net)
        print(y)

