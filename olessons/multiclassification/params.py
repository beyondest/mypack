import numpy as np
import torch
import sklearn.datasets as skd
import sys
sys.path.append('../../utils_network/')
from utils_network.data import *
from utils_network.mymodel import *
import pandas as pd

dataset_path='../../datasets/mnist'

img_path='./test.png'
epochs = 100
learning_rate = 0.01
batch_size = 100

weights_save_path='./weights1.pth'