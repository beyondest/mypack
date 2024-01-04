
import sys
sys.path.append('../../')
from params import *
from utils_network.data import *
from torchvision import transforms,datasets
from torch.utils.data import Dataset,DataLoader
from utils_network.actions import *

import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



model = mobilenet_v2(num_classes=2)

val_dog_folder = '/mnt/d/datasets/petimages/val/Dog'


if 0:
    predict_classification(model,
                       val_trans,
                       './res',
                       trained_weights_path,
                       class_yaml_path,
                       if_draw_and_show_result=True)

validation(model,val_trans,5,val_root_path,trained_weights_path)



