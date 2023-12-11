import os
import numpy as np
from skimage import io, util, img_as_ubyte
from skimage.util import random_noise
import sys
sys.path.append('../os_op')
sys.path.append('../utils_network')
import os_op.os_operation as oso
import img_operation as imo
import cv2
from utils_network.data import *


input_path = "/mnt/d/datasets/for_noised"
output_path="/mnt/d/datasets/noised_already"

def add_noise(img_bgr:np.ndarray, noise_type='gaussian', seed:int = 10)->np.ndarray:
    
    
    img= cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
    if noise_type == 'gaussian':
        noisy_image = random_noise(img, mode='gaussian', seed=seed)
    elif noise_type == 'salt-and-pepper':
        noisy_image = random_noise(img, mode='s&p', seed=seed)
    elif noise_type == 'speckle':
        noisy_image = random_noise(img, mode='speckle', seed=seed)
    else:
        raise ValueError("Unsupported noise type")
    
    #NOTICE skimage will turn image to normalized form!!!
    noisy_image = img_as_ubyte(noisy_image)
    
    noisy_image_bgr = cv2.cvtColor(noisy_image,cv2.COLOR_RGB2BGR)
    return noisy_image_bgr

def add_noise2(abs_path:str,outabs_path:str,noise_type:str ="gaussian",seed:int = 10 )->None:
    img = cv2.imread(abs_path)
    noisy_img_bgr = add_noise(img,noise_type=noise_type,seed=seed)
    cv2.imwrite(outabs_path,noisy_img_bgr)
    



if __name__ == "__main__":
    
    oso.traverse(input_path,output_path,"png",add_noise2,seed = 11)
    def show2(in_path,out_path= None):
        img=cv2.imread(in_path)
        imo.cvshow(img)
    oso.traverse(output_path,None,'png',show2)