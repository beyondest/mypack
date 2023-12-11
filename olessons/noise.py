import numpy as np
import sympy
import scipy
import random 
import time
import matplotlib.pyplot as plt
import math
import scipy
import sys
sys.path.append('../utils_network/')
sys.path.append('../os_op')
sys.path.append('../img')
import os_op.os_operation as oso
import img.img_operation as imo
import cv2
from utils_network.data import *


input_img_path=  "/mnt/d/datasets/for_noised"
output_img_path = "/mnt/d/datasets/noised_already"



if __name__ == "__main__":
    oso.traverse(input_img_path,output_img_path,None,imo.add_noise2)
    
    
    
    oso.traverse(output_img_path,None,None,imo.cvshow2)
    
    