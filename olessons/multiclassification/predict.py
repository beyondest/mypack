import numpy as np
import torch
import sklearn.datasets as skd
import sys
sys.path.append('../..')
from utils_network.data import *
from utils_network.mymodel import *
import pandas as pd
import cv2
import torch
from torchvision import transforms
from PIL import Image
from olessons.multiclassification.params import *
from img.img_operation import *
from utils_network.actions import *
test_mnist_img = 'out2/mnist_0.jpg'
my_test_img = 'test.png'

model = lenet5() 

img = cv2.imread(my_test_img)
def write_img_trans(img:np.ndarray)->np.ndarray:
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #_,img = cv2.threshold(img,0,255,cv2.THRESH_OTSU)
    
    img = cv2.bitwise_not(img)
    img = cv2.dilate(img,np.ones((int(img.shape[0]/80),int(img.shape[1]/80))),iterations=3)
    
    img = cv2.resize(img,(28,28))
    _,img = cv2.threshold(img,0,255,cv2.THRESH_OTSU)
    return img

def show_train_trans(abs_path):
    img = Image.open(abs_path)
    img = cv_trans_train(img)


#oso.traverse('out2',fmt='jpg',deal_func=show_train_trans)

if 1:
    
    predict_classification(model,
                        val_trans,
                        'test',
                        trained2_weights_path,
                        class_yaml_path,
                        'jpg',
                        custom_trans_cv=write_img_trans,
                        if_show=True)
    
    
    
    
    
    

        
