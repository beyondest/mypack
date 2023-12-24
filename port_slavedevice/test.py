import sys
sys.path.append('..')
import img.img_operation as imo

import cv2
from os_op.thread_op import *

def fu(canvas:imo.Img.canvas,x_list):
    img = canvas.img.copy()
    
    img = imo.add_text(img,'fs',x_list[0])
    cv2.imshow('ff',img)
    

def deinit():
    cv2.destroyAllWindows()

def add(x_list):
     x_list[0]+=1
    
canvas = imo.Img.canvas((100,100))
x_list = [2]

task1 = task(0.5,
             for_circle_func=add,
             params_for_circle_func=[x_list]
             )

keyboard_control_task([task1],main_func=fu,main_func_params=[canvas,x_list])
