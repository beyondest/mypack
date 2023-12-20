import numpy as np
import torch
import sklearn.datasets as skd
import sys
sys.path.append('../../utils_network/')
from utils_network.data import *
from utils_network.mymodel import *
import pandas as pd
from torchvision import transforms,datasets


dataset_path='./data'

img_path='./test.png'
epochs = 100
learning_rate = 0.001
batch_size = 20

weights_save_path='./weights2'
trained_weights_path = './weights2/weights.0.92.1.pth'
class_yaml_path = './class.yaml'


train_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomRotation(45)
])
val_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(),
    transforms.Resize((28,28))
])

