import onnx
import onnxruntime
import onnxruntime.quantization as oq

from torch.utils.data import DataLoader
from torchvision import datasets
import sys
sys.path.append('../..')
from params import *
from utils_network.data import *




dataset =datasets.ImageFolder(val_root_path,val_trans)
dataset = get_subset(dataset,[500,1000])
dataloader = DataLoader(dataset,1,False)

dreader = Dataloader_CalibrationDataReader(dataloader)


if 0:
    oq.quant_pre_process(fixed_model_path,
                    infer_model_path,
                    skip_optimization=False,
                    skip_onnx_shape=False,              # works best for non-Transformer
                    skip_symbolic_shape=False,           # works best for Transformer
                    auto_merge=False,
                    verbose=True,
                    save_as_external_data=False,
                    all_tensors_to_one_file=False,
                    external_data_location='./external_data'
                    )

if 1:
    
    oq.quantize_dynamic(infer_model_path,
                     dynamic_model_path,
                     op_types_to_quantize=None, # ['Conov',...]
                     per_channel=False,          # can improve accuracy
                     reduce_range=True,         # use for U8S8 format on non-VNNI machine and for per_channel_True on non-VNNI machine
                     weight_type=oq.QuantType.QUInt8,
                     nodes_to_quantize=None,    # [Node1, Node2 ,...]
                     nodes_to_exclude=None,     # [Node1, Node2, ...]
                     use_external_data_format=False # if save tensors in other files and not in model file
                    )

if 0:
    
    oq.quantize_static(infer_model_path,
                    static_model_path,
                    calibration_data_reader=dreader,    # use Trans to get datareader from dataloader
                    quant_format=oq.QuantFormat.QDQ,    # QDQ or QO, QO is larger ?
                    op_types_to_quantize=None,  
                    per_channel=True,
                    reduce_range=True,
                    activation_type=oq.QuantType.QUInt8,        #U8S8 run fastest in my CPU
                    weight_type=oq.QuantType.QInt8,
                    nodes_to_quantize=None,
                    nodes_to_exclude=None,
                    use_external_data_format=None,                      
                    calibrate_method=oq.CalibrationMethod.MinMax        # define scaler by max and min of calibration dataset
                    )



