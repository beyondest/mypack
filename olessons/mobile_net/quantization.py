import onnx
import onnxruntime
import onnxruntime.quantization as oq

from torch.utils.data import DataLoader
from torchvision import datasets
import sys
sys.path.append('../..')
from params import *
from utils_network.data import *
from utils_network.actions import *



dataset =datasets.ImageFolder(val_root_path,val_trans)
dataset = get_subset(dataset,[500,1000])
dataloader = DataLoader(dataset,1,False)

if 1:
    Onnx_Processer.end_to_end_opt_qua(ori_onnx_path,
                                  dataloader,
                                  if_save_tmp_refenrece=True
                                  )

