import numpy as np
import torch
import sklearn.datasets as skd
import sys
sys.path.append('../../utils_network/')
from utils_network.data import *
from utils_network.mymodel import *
import pandas as pd
import cv2
import torch
from torchvision import transforms
from PIL import Image
from olessons.multiclassification.params import *




def show(img):
    cv2.namedWindow("win1",cv2.WINDOW_AUTOSIZE)
    cv2.imshow("win1",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def load_img(img_path):
    ''''''
    img=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    
    
    img=cv2.bitwise_not(img)
    dilate_k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    img=cv2.dilate(img,dilate_k,iterations=11)
    
    
    img=cv2.resize(img,(28,28),interpolation=cv2.INTER_LINEAR)
    
    return img

model = lenet5()  
model.load_state_dict(torch.load(weights_save_path))
model.eval()
ori_img=cv2.imread('./test.png')
img=load_img('./test.png')


preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((28, 28)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

input_image = preprocess(img).unsqueeze(0)  


with torch.no_grad():
    output = model(input_image)


_, predicted_class = torch.max(output, 1)


img2=ori_img.copy()
cv2.putText(img2,f'The predicted class is: {predicted_class.item()}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
show(img2)
