import onnxruntime
import numpy as np
import sys

import cv2
sys.path.append('../..')
from os_op.decorator import *
from params import *
from utils_network.actions import *
from torch.utils.data import Subset
onnx_filename = dynamic_model_path


ori_img = cv2.imread('res/cat.1.jpg')

mode = 1



final_input = val_trans(ori_img)
final_input = torch.unsqueeze(final_input,dim=0)
final_input = final_input.numpy()

onnx_engine = Onnx_Engine(onnx_filename)


if mode == 0:
    
    out,t = onnx_engine.run(None,{'input':final_input})


    print(f'{t:6f}')
    p,i = trans_logits_in_batch_to_result(out[0])
    print(p,i)
    
    
else:
    
    dataset = datasets.ImageFolder(val_root_path,val_trans)
    subset = get_subset(dataset,[500,1000])
    data_loader = DataLoader(subset,
                             1,
                             True)
    
    onnx_engine.eval_run_node0(data_loader,
                               'input')
    

