import numpy as np
import torch
import sklearn.datasets as skd
import sys
sys.path.append('../../utils_network/')
from utils_network.data import *
from utils_network.mymodel import *
import pandas as pd
from torchvision import transforms,datasets
from img.img_operation import *


dataset_path='./data'

img_path='./test.png'
epochs = 5
learning_rate = 0.001
batch_size = 30

weights_save_path='./weights2'
trained_weights_path = './weights2/weights.0.92.1.pth'
trained2_weights_path = './weights2/best71.4.pth'

class_yaml_path = './class.yaml'


show_trans = transforms.Compose([

    transforms.RandomAffine(degrees=0,translate=(0.2,0.1),scale=(1,2),shear=45)
])

def cv_trans_train(img:np.ndarray):

    img = add_noise_circle(img,circle_radius_ratio=1/6,noise_probability=1/3)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,img = cv2.threshold(img,0,255,cv2.THRESH_OTSU)
    return img
    
def cv_trans_val(img:np.ndarray):

    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,img = cv2.threshold(img,0,255,cv2.THRESH_OTSU)
    return img

# Notice : original mnist dataset img is grayscale!!!

train_trans = transforms.Compose([
    transforms.Resize((40,35)),
    transforms.RandomCrop(28),
    transforms.RandomAffine(degrees=0,translate=(0.2,0.1),scale=(1,1.4),shear=30),
    PIL_img_transform(cv_trans_train,'gray'),
    transforms.ToTensor()
])

val_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.ToPILImage(),
    PIL_img_transform(cv_trans_val,'gray'),
    transforms.ToTensor(),
    transforms.Resize((28,28))
])

