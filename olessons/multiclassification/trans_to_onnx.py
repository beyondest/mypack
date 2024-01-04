import torch.onnx
import torch
import torch.onnx
from torch.quantization import quantize_dynamic, quantize_qat

import sys
sys.path.append('../..')
from utils_network.mymodel import *
from params import *


model = lenet5()
model.load_state_dict(torch.load(trained2_weights_path,map_location=device))
model.eval()
dummy_input = torch.randn((1,1,28,28))

input_names = ['input']
output_names = ['output']
onnx_name = 'best71.onnx'
dynamic_axe_name = 'batch_size'
dynamic_axe_input_id = 0
dynamic_axe_output_id = 0

dynamic_axes = {input_names[0]: {dynamic_axe_input_id: dynamic_axe_name}, 
                output_names[0]: {dynamic_axe_output_id: dynamic_axe_name}}


# quatized model, but notice not all platforms onnx run will support this, so you need to add ATEN_FALLBACK 
q_model = quantize_dynamic(model,dtype=torch.qint8)

torch.onnx.export(model=model,              #model to trans
                  args=dummy_input,         #dummy input to infer shape
                  f=onnx_name,              #output onnx name 
                  verbose=True,             #if show verbose information in console
                  export_params=True,       #if save present weights to onnx
                  input_names=input_names,  #input names list,its length depends on how many input your model have
                  output_names=output_names,#output names list
                  training=torch.onnx.TrainingMode.EVAL,        #EVAL or TRAINING or Preserve(depends on if you specify model.eval or model.train)
                  operator_export_type=torch.onnx.OperatorExportTypes.ONNX,   #ONNX or this, ATEN means array tensor library of Pytorch
                  opset_version=17,         #7<thix<17
                  do_constant_folding=True, #True
                  dynamic_axes = dynamic_axes,      #specify which axe is dynamic
                  keep_initializers_as_inputs=False,#if True, then you can change weights of onnx if model is same   
                  custom_opsets=None,               #custom operation, such as lambda x:abs(x),of course not so simple, you have to register to pytorch if you want to use custom op
                  export_modules_as_functions=False,#False
                  autograd_inlining=True)           #True

print('*****************************success***************************')

