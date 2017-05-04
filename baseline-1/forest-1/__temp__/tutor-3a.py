# cifar 10 example
# http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# https://github.com/kuangliu/pytorch-cifar/tree/master/models

import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torch.optim as optim



import matplotlib.pyplot as plt
import numpy as np
import cv2

import os
from timeit import default_timer as timer   #ubuntu:  default_timer = time.time,  seconds

# Cifar10 Dataset -------------------------------------------------------------------------------
import torchvision.transforms as transforms
transform = transforms.Compose([
                transforms.ToTensor(),    # this is in range of [0,1]
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # range in [-1,1]
                    # Normalize(mean, std): channel = (channel - mean) / std
            ])

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
train_dataset = torchvision.datasets.CIFAR10(root='/root/share/project/pytorch/data/cifar10',
                            train=True,
                            transform=transform,
                            download=True)

test_dataset = torchvision.datasets.CIFAR10(root='/root/share/project/pytorch/data/cifar10',
                            train=False,
                            transform=transform,
                            download=True)

#use default loader
# see http://pytorch.org/docs/data.html
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=2)  ##shuffle=True
test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=2)
train_iter   = iter(train_loader)
test_iter    = iter(test_loader )

#<todo> use custom loader




def im_show(name, image, resize=1):
    H,W = image.shape[0:2]
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image.astype(np.uint8))
    cv2.resizeWindow(name, round(resize*W), round(resize*H))


#convert to images---------------------------------------------------------------
DATA_DIR = '/root/share/project/pytorch/data/cifar10/rgb'

if 0:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DATA_DIR+'/test',  exist_ok=True)
    os.makedirs(DATA_DIR+'/train', exist_ok=True)
    for n in range(10):
        os.makedirs(DATA_DIR+'/test/%d'%n,  exist_ok=True)
        os.makedirs(DATA_DIR+'/train/%d'%n, exist_ok=True)

if 0:
    counter=0
    for i, data in enumerate(train_loader, 0):
        images, labels = data
        num = len(images)

        for n in range(num):
            label=labels[n]
            print('label=%d : %s'%(label,classes[label]))
            img_file = DATA_DIR+'/train/%d/%08d.png'%(label,counter)
            counter+=1

            img = np.transpose(images[n].numpy(), (1, 2, 0))
            img = (img*0.5 + 0.5)*255
            img =  cv2.cvtColor( img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            cv2.imwrite(img_file,img)

            # im_show('img',img,8)
            # cv2.waitKey(1)


if 0:
    counter=0
    for i, data in enumerate(test_loader, 0):
        images, labels = data
        num = len(images)

        for n in range(num):
            label=labels[n]
            print('label=%d : %s'%(label,classes[label]))
            img_file = DATA_DIR+'/test/%d/%08d.png'%(label,counter)
            counter+=1

            img = np.transpose(images[n].numpy(), (1, 2, 0))
            img = (img*0.5 + 0.5)*255
            img =  cv2.cvtColor( img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            cv2.imwrite(img_file,img)

            # im_show('img',img,8)
            # cv2.waitKey(1)


##exit(0)
##--------------------------------------------------------------------------------








# check the data
if 0:

    # get some random training images
    images, labels = train_iter.next()
    num = len(images)
    for n in range(num):
        label=labels[n]
        print('label=%d : %s'%(label,classes[label]))

        img = np.transpose(images[n].numpy(), (1, 2, 0))
        img = (img*0.5 + 0.5)*255
        img =  cv2.cvtColor( img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        im_show('img',img,8)
        cv2.waitKey(0)



# CNN Model  ----------------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
print (net)


# Optimiser  ----------------------------------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

model_dir ='./snap'

#training cpu
if 0:
    start = timer()

    i_print=100
    for epoch in range(3):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % i_print == i_print-1:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / i_print))
                running_loss = 0.0

    print('Finished Training')
    end = timer()
    print ('cpu: end-start = %0.2f  min'%((end - start)/60))

    #saving
    # http://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytroch
    torch.save(net.state_dict(), model_dir)

    '''
    [1,   100] loss: 2.302
    [1,   200] loss: 2.303
    [1,   300] loss: 2.300
    [1,   400] loss: 2.299
    [1,   500] loss: 2.296 
    
    
    [3,  1300] loss: 1.549
    [3,  1400] loss: 1.556
    [3,  1500] loss: 1.538
    Finished Training
    cpu: end-start = 1.79  min    
    '''


#training gpu
if 0:
    net.cuda()

    start = timer()
    i_print=100
    for epoch in range(20):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % i_print == i_print-1:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / i_print))
                running_loss = 0.0

    print('Finished Training')
    end = timer()
    print ('gpu: end-start = %0.2f  min'%((end - start)/60))

    #saving
    # http://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytroch
    net.cpu()
    torch.save(net.state_dict(), model_dir)

    '''
    [1,   100] loss: 2.305
    [1,   200] loss: 2.301
    [1,   300] loss: 2.301
    [1,   400] loss: 2.301
    [1,   500] loss: 2.299
    
    
    [3,  1300] loss: 1.575
    [3,  1400] loss: 1.575
    [3,  1500] loss: 1.547
    Finished Training
    gpu: end-start = 0.24  min
    '''


# Testing  ----------------------------------------------------------------------------
net.load_state_dict(torch.load(model_dir))

# test a few ...
images, labels = test_iter.next()
outputs = net(Variable(images))
maxs, indices = torch.max(outputs.data, 1)

maxs    = maxs.numpy().reshape(-1)
indices = indices.numpy().reshape(-1)

num = len(images)
for n in range(num):
    print('score')
    print(outputs[n])

    max=maxs[n]
    index=indices[n]
    print('max,index = %f, %d : %s'%(max,index,classes[index]))
    label=labels[n]
    print('label=%d : %s'%(label,classes[label]))
    print('')

    img = np.transpose(images[n].numpy(), (1, 2, 0))
    img = (img*0.5 + 0.5)*255
    img =  cv2.cvtColor( img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    im_show('img',img,8)
    cv2.waitKey(1)


# test all ...
correct = 0
total = 0

for data in test_loader:
    images, labels = data
    outputs = net(Variable(images))
    maxs, indices = torch.max(outputs.data, 1)

    total   += labels.size(0)
    correct += (indices == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

'''
reference results from tutorial web_page:
Accuracy of the network on the 10000 test images: 55 %
'''

# BUG in python 3.5 !!!!!!!!!!
#
# https://discuss.pytorch.org/t/train-opennmt-and-typeerror-nonetype-object-is-not-callable/1478
#  I think might be seeing a bug in python 3.5 weakref7 that occurs during shutdown.
#
#    Exception ignored in: <function WeakValueDictionary.__init__.<locals>.remove at 0x7f8caa936950>
#
#  note: no bug in python 2.7 or 3.6
