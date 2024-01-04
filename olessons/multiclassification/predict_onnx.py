import onnxruntime
import numpy as np
import sys

import cv2
sys.path.append('../..')
from os_op.decorator import *
from params import *
onnx_filename = 'best71.onnx'

custom_session_options = onnxruntime.SessionOptions()
custom_session_options.enable_profiling = False          #enable or disable profiling of model
#custom_session_options.execution_mode =onnxruntime.ExecutionMode.ORT_PARALLEL       #ORT_PARALLEL | ORT_SEQUENTIAL
custom_session_options.add_session_config_entry('session.load_model_format', 'ONNX') # or 'ORT'
#custom_session_options.inter_op_num_threads = 2                                     #default is 0
#custom_session_options.intra_op_num_threads = 2                                     #default is 0
#custom_session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL # DISABLE_ALL |ENABLE_BASIC |ENABLE_EXTENDED |ENABLE ALL

custom_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']       # if gpu, cuda first, or will use cpu


class Standard_Data:
    def __init__(self) -> None:
        self.result = 0
        
    def save_results(self,results:np.ndarray):
        self.result = results
user_data = Standard_Data()      
def standard_callback(results:np.ndarray, user_data:Standard_Data,error:str):
    if error:
        print(error)
    else:
        user_data.save_results(results)


ort_session = onnxruntime.InferenceSession(onnx_filename,
                                           sess_options=custom_session_options,
                                           providers=custom_providers)


ori_img = cv2.imread('test.png')



idx_to_class = Data.get_file_info_from_yaml(class_yaml_path)


#
img = write_img_trans(ori_img)
input_tensor = val_trans(img)
input_tensor = torch.unsqueeze(input_tensor,0)
#


print('model running')

@timing(circle_times=10000,if_show_total=True)
def run():


    '''img2 = cv2.imread('3.png')
    img2 = write_img_trans(img2)
    input_tensor2 = val_trans(img2)
    input_tensor2 = torch.unsqueeze(input_tensor2,0)
    input_all = torch.cat((input_tensor,input_tensor2),dim=0)
'''

    outputs = ort_session.run(['output'], {'input': input_tensor.numpy()})
    #outputs = ort_session.run_async(None,{'input':input_tensor.numpy()},callback=standard_callback,user_data=user_data)
    return outputs


result,elapesd_time = run()
print(f'avg_spending: {elapesd_time:2f}')

logits = result[0]

softmax_result = torch.softmax((torch.from_numpy(logits)),dim=1)
max_probability = torch.max(softmax_result,dim=1).values
max_index = np.argmax(logits,axis=1)
print(max_probability)
print(max_index)



