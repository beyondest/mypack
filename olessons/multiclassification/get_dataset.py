import numpy as np
import torch
import sklearn.datasets as skd
import sys
sys.path.append('../../utils_network/')
from utils_network.data import *
from utils_network.mymodel import *
import pandas as pd
import torchvision.transforms as tvt
import torchvision
from params import *

train_dataset=torchvision.datasets.MNIST(root=dataset_path,train=True,download=True)
test_dataset= torchvision.datasets.MNIST(root=dataset_path,train=False,download=True)


print(type(train_dataset))

