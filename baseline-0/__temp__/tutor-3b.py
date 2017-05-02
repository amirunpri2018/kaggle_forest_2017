#  what is pinned memory
#       https://devblogs.nvidia.com/parallelforall/how-optimize-data-transfers-cuda-fortran/
#


# pytorch custom data
#       http://stackoverflow.com/questions/43441673/trying-to-load-a-custom-dataset-in-pytorch
#       http://forums.fast.ai/t/how-do-you-use-custom-dataset-with-pytorch/2040
#       https://discuss.pytorch.org/t/questions-about-imagefolder/774/6


## https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252/3
## https://discuss.pytorch.org/t/questions-about-imagefolder/774
## https://github.com/pytorch/tutorials/issues/78
## https://www.kaggle.com/mratsim/planet-understanding-the-amazon-from-space/starting-kit-for-pytorch-deep-learning
## http://pytorch.org/docs/_modules/torch/utils/data/dataset.html
## https://www.kaggle.com/mratsim/planet-understanding-the-amazon-from-space/starting-kit-for-pytorch-deep-learning/notebook
## https://devhub.io/repos/pytorch-vision
## https://github.com/ClementPinard/FlowNetPytorch/blob/master/balancedsampler.py


#my libs
from net.common import *

#pytorch
import torch
import torchvision.transforms as transforms

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler



## helper functions ----------------------------------------------



