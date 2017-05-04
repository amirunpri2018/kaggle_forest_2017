## https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/00%20-%20PyTorch%20Basics/main.py
## http://qiita.com/miyamotok0105/items/1fd1d5c3532b174720cd

## http://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py

import torch
import torchvision
import torch.nn as nn

import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable

import numpy as np
import os

#======================= Basic autograd example 1 =======================#
def run_exp1():

    # Create tensors.
    x = Variable(torch.Tensor([6]), requires_grad=True)
    w = Variable(torch.Tensor([2]), requires_grad=True)
    b = Variable(torch.Tensor([3]), requires_grad=True)

    # Build a computational graph.
    y = w * x + b    # y = 2 * x + 3
    print (y.creator)

    # Compute gradients.
    y.backward()

    # Print out the gradients.
    print('x.grad=',x.grad)    # x.grad = 2
    print('w.grad=',w.grad)    # w.grad = 1
    print('b.grad=',b.grad)    # b.grad = 1


    pass
#======================== Basic autograd example 2 =======================#
def run_exp2():

    x=Variable(torch.randn(5,3))
    y=Variable(torch.randn(5,2))

    # Build a linear layer.
    linear = nn.Linear(3, 2)
    print ('w: ', linear.weight)
    print ('b: ', linear.bias)

    # Build Loss and Optimizer.
    mse = nn.MSELoss()

    # Forward propagation.
    est = linear(x)
    print ('est: ', est)


    # Compute cost.
    loss = mse(est, y)
    print('cost: ', loss.data[0])


    # Backpropagation.
    loss.backward(retain_variables=True)

    print ('dw: ', linear.weight.grad)
    print ('db: ', linear.bias.grad)


    ## Optimization (gradient descent).
    optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)
    for iter in range(10):
        loss.backward()
        optimizer.step()

        est = linear(x)
        loss = mse(est, y)
        print('cost: ', loss.data[0])

    pass








# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_exp2()