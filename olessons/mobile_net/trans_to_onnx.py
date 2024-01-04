import torch.onnx
import torch
import torch.onnx

import sys
sys.path.append('../..')
from utils_network.mymodel import *
from params import *
from utils_network.data import *

model = mobilenet_v2(num_classes=2)
dummy = torch.randn((1,3,224,224))
Data.save_model_to_onnx(model,
                        'static83.onnx',
                        dummy,
                        trained_weights_path,
                        if_dynamic_batch_size=False)

