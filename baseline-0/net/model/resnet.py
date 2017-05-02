#  https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
#  https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py


import torch
import torch.nn as nn
import torch.nn.functional as F

from net.common import *
from net.utility.tool import *




class Basic(nn.Module):

    def __init__(self, in_channels, channels, expansion=1, stride=1):
        super(Basic, self).__init__()
        out_channels = expansion*channels

        self.conv1 = nn.Conv2d(in_channels, channels,  kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, out_channels, kernel_size=3, stride=1,      padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.in_channels  = in_channels
        self.out_channels = out_channels


    def forward(self, x):
        out =  F.relu(self.bn1(self.conv1(x  )))
        out =  F.relu(self.bn2(self.conv2(out)))
        out += self.shortcut(x) if self.shortcut is not None else x
        out =  F.relu(out)
        ## out = 0*out  ##debug
        return out



class Bottleneck(nn.Module):

    def __init__(self, in_channels, channels, expansion=4, stride=1):
        super(Bottleneck, self).__init__()
        out_channels = expansion*channels

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, out_channels, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_channels)

        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.in_channels  = in_channels
        self.out_channels = out_channels


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x  )))
        out = F.relu(self.bn2(self.conv2(out)))
        out =        self.bn3(self.conv3(out))
        out += self.shortcut(x) if self.shortcut is not None else x
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def make_layer(self, block, in_channels, channels, num_blocks, stride):

        layers = []
        strides = [stride] + [1]*(num_blocks-1)
        for stride in strides:
            b = block(in_channels, channels, stride)
            in_channels = b.out_channels
            layers.append(b)

        return nn.Sequential(*layers), in_channels


    def __init__(self, block, in_channels, channels, num_blocks, num_classes=10):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels[0])
        in_channels = channels[0]

        self.layer1, in_channels = self.make_layer(block, in_channels, channels[1], num_blocks[1], stride=1)
        self.layer2, in_channels = self.make_layer(block, in_channels, channels[2], num_blocks[2], stride=2)
        self.layer3, in_channels = self.make_layer(block, in_channels, channels[3], num_blocks[3], stride=2)
        self.layer4, in_channels = self.make_layer(block, in_channels, channels[4], num_blocks[4], stride=2)
        self.linear = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, kernel_size=out.size()[2:])  #4
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(in_channels=3, num_classes=10):
    return ResNet(Basic,      in_channels=in_channels, channels=[32,32,64,128,256], num_blocks=[0,2,2, 2,2],num_classes=num_classes)

def ResNet18_1(in_channels=3, num_classes=10):
    return ResNet(Basic,      in_channels=in_channels, channels=[64,64,128,256,512], num_blocks=[0,2,2, 2,2],num_classes=num_classes)

def ResNet34(in_channels=3, num_classes=10):
    return ResNet(Basic,      in_channels=in_channels, channels=[64,64,128,256,512], num_blocks=[0,3,4, 6,3],num_classes=num_classes)

def ResNet50(in_channels=3, num_classes=10):
    return ResNet(Bottleneck, in_channels=in_channels, channels=[64,64,128,256,512], num_blocks=[0,3,4, 6,3],num_classes=num_classes)

def ResNet101(in_channels=3, num_classes=10):
    return ResNet(Bottleneck, in_channels=in_channels, channels=[64,64,128,256,512], num_blocks=[0,3,4,23,3],num_classes=num_classes)

def ResNet152(in_channels=3, num_classes=10):
    return ResNet(Bottleneck, in_channels=in_channels, channels=[64,64,128,256,512], num_blocks=[0,3,8,36,3],num_classes=num_classes)




# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    # https://discuss.pytorch.org/t/print-autograd-graph/692/8
    inputs = torch.randn(32,3,32,32)

    #check Basic(), Bottleneck()
    if 0:
        in_channels  = inputs.size()[1]
        out_channels = 5  ##5
        #b = Basic(in_channels,out_channels)
        b = Bottleneck(in_channels,out_channels)
        y = b.forward(Variable(inputs))
        dot = make_dot(y)
        dot.view()
        print(b)
        print(y)


    if 0:
        net = ResNet18()
        x = Variable(inputs)

        start = timer()
        y = net.forward(x)
        end = timer()
        print ('end-start=%0.0f  ms'%((end - start)*1000))

        dot = make_dot(y)
        dot.view()
        print(net)
        print(y)



    if 1:
        net = ResNet18().cuda()
        x = Variable(inputs).cuda()

        start = timer()
        y = net.forward(x)
        end = timer()
        print ('cuda(): end-start=%0.0f  ms'%((end - start)*1000))

        #dot = make_dot(y)
        #dot.view()
        print(net)
        print(y)

        save_dot(y,
                    { x: 'x', y: 'y' },
                    open('/root/share/project/pytorch/build/cifar-0/net/model/mlp.dot', 'w'))

