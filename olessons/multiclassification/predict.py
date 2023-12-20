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

from utils_network.actions import *



model = lenet5()  
predict_classification(model,val_trans,'./out',trained_weights_path,class_yaml_path,'jpg')

