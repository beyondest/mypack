import torch
import sys
sys.path.append('..')
import os_op.os_operation as oso
import os
import time

from torchvision import transforms,datasets
import cv2
from utils_network.data import *
import onnx
import onnxruntime
from os_op.decorator import *
from onnxruntime.tools.convert_onnx_models_to_ort import convert_onnx_models_to_ort
from pathlib import Path
def train_classification(   model:torch.nn.Module,
                            train_dataloader,
                            val_dataloader,
                            device:None,
                            epochs:int,
                            criterion,
                            optimizer,
                            weights_save_path:str|None,
                            save_interval:int = 1,
                            show_step_interval:int=10,
                            show_epoch_interval:int=2,
                            log_folder_path:str = './log',
                            log_step_interval:int = 10,
                            log_epoch_interval:int = 1
                            ):
    '''
    if show_step_interval<0, will not show step
    '''
    from torch.utils.tensorboard import SummaryWriter
    if not os.path.exists(log_folder_path):
        os.mkdir(log_folder_path)
    writer = SummaryWriter(log_dir=log_folder_path)
    oso.get_current_datetime(True)
    t1 = time.perf_counter()
    print(f'All {epochs} epochs to run, each epoch train steps:{len(train_dataloader)}')
    step_show =True
    epoch_show = True
    log_step_write = True
    log_epoch_write = True
    
    if show_step_interval <0:
        step_show = False
    if show_epoch_interval<0:
        epoch_show = False
        
    if log_step_interval<0:
        log_step_write = False
    if log_epoch_interval < 0:
        log_epoch_write = False 
        
    val_step_nums = len(val_dataloader)
    train_step_nums = len(train_dataloader)
    save_path = os.path.splitext(weights_save_path)[0]
    
    
    best_accuracy = 0
    epoch_loss = 0
    all_loss = 0
    
    for epoch in range(epochs):
        step = 0
        step_loss = 0
        epoch_loss = 0
        
        
        model.train()
        for step,sample in enumerate(train_dataloader):
            X,y = sample
            model.zero_grad()
            logits = model(X.to(device))
            loss = criterion(logits,y.to(device))
            loss.backward()
            optimizer.step()
            
            step_loss+= loss.item()
            
            
            if step_show:
                if step%show_step_interval ==0:
                    print(f'step:{step}/{train_step_nums}   step_loss:{loss.item():.2f}')
            if log_step_write:
                if step%log_step_interval == 0:
                    writer.add_scalar('avg_step_loss',step_loss/(step+1),epoch * train_step_nums+step)

        
        
        model.eval()
        with torch.no_grad():
            right_nums = 0
            sample_nums = 0
            for X,y in val_dataloader:
                logits = model(X.to(device))
                
                #e.g.:logits.shape = (20,2)=(batchsize,class), torch.max(logits).shape = (2,20),[0] is value, [1].shape = (10,1),[1] =[0,1,...] 
                #use torch.max on logits without softmax is same as torch.max(softmax(logits),dim=1)[1]
                predict = torch.max(logits,dim=1)[1]
                #caclulate
                right_nums+=(predict == y.to(device)).sum().item()
                sample_nums += y.size(0)
            accuracy =right_nums/sample_nums
            
        if accuracy>best_accuracy:
            best_accuracy = accuracy
        epoch_loss+=step_loss 
        all_loss+=epoch_loss
        
        
        if epoch_show:   
            if epoch%show_epoch_interval== 0:
                oso.get_current_datetime(True)
                print(f"epoch:{epoch+1}/{epochs}    epoch_loss:{epoch_loss:.2f}         accuracy:{accuracy:.2f}         best_accuracy:{best_accuracy:.2f}")
        
        if log_epoch_write:
            if epoch%log_epoch_interval == 0:
                writer.add_scalar('avg_epoch_loss',all_loss/(epoch+1),epoch)
        writer.add_scalar('epoch_accuracy',accuracy,epoch)
        
        
        if weights_save_path is not None:
            if epoch%save_interval == 0:
                name = f'weights.{accuracy:.2f}.{epoch}.pth'
                current_save_path = os.path.join(save_path,name)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                
                print(f'model.state_dict save to {current_save_path}')
                torch.save(model.state_dict(),current_save_path)
          
    oso.get_current_datetime(True)
    t2 = time.perf_counter()
    t = t2-t1
    print(f"Training over,Spend time:{t:.2f}s")
    print(f"Log saved to folder: {log_folder_path}")
    print(f'Weights saved to folder: {save_path}')
    print(f"Best accuracy : {best_accuracy} ")
    
    
def predict_classification(model:torch.nn.Module,
                           trans,
                           img_or_folder_path:str,
                           weights_path:str,
                           class_yaml_path:str,
                           fmt:str = 'jpg',
                           custom_trans_cv:None = None,
                           if_show: bool = False,
                           if_draw_and_show_result:bool = False,
                           if_print:bool = True
                           ):
    """Predict single or a folder of images , notice that if input is grayscale, then you have to transform it in trans!!!

    Args:
        model (torch.nn.Module): _description_
        trans (_type_): _description_
        img_or_folder_path (str): _description_
        weights_path (str): _description_
        class_yaml_path (str): _description_
        fmt (str, optional): _description_. Defaults to 'jpg'.
        if_cvt_rgb (bool, optional): _description_. Defaults to True.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    idx_to_class = Data.get_file_info_from_yaml(class_yaml_path)
    model.load_state_dict(torch.load(weights_path,map_location=device))
    model.eval()
    suffix = os.path.splitext(img_or_folder_path)[-1]
    if suffix != '':
        mode = 'single_img'
    else:
        mode = 'multi_img'
    
    def single_predic(img_or_folder_path):
        img_ori = cv2.imread(img_or_folder_path)
        if custom_trans_cv is not None:
            img = custom_trans_cv(img_ori)
        else:
            img = img_ori   
            
        if if_show:
            imo.cvshow(img,f'{img_or_folder_path}')
        input_tensor = trans(img).unsqueeze(0).to(device)

        
        with torch.no_grad():
            t1 = time.perf_counter()
            logits = model(input_tensor)
            t2 = time.perf_counter()
            t = t2-t1
            probablility_tensor = torch.nn.functional.softmax(logits,dim=1)[0]
            probablility = torch.max(probablility_tensor).item()
            predict_class_id = torch.argmax(probablility_tensor).item()
            predict_class = idx_to_class[predict_class_id]
            if if_print:
                print(f'{img_or_folder_path} is {predict_class}, with probablity {probablility:.2f}    spend_time: {t:6f}')

        if if_draw_and_show_result:
            imo.add_text(img_ori,f'class:{predict_class} | probability: {probablility}',0,color=(0,255,0))
            imo.cvshow(img_ori,'result')
            
    if mode == 'single_img':
        single_predic(img_or_folder_path)
        

        
    else:
        oso.traverse(img_or_folder_path,None,deal_func=single_predic,fmt=fmt)
        
        
        
def validation(model:torch.nn.Module,
               trans,   
               batchsize,
               img_root_folder:str,
               weights_path:str
               ):
    
    val_dataset = datasets.ImageFolder(img_root_folder,trans)
    val_dataloader = DataLoader(val_dataset,
                                batchsize,
                                shuffle=True,
                                num_workers=1)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(weights_path,map_location=device))
    model.eval()
    
    
    with torch.no_grad():
        right_nums = 0
        sample_nums = 0
        for i,sample in enumerate(val_dataloader):
            X,y = sample
            t1 = time.perf_counter()
            logits = model(X.to(device))
            t2 = time.perf_counter()
            t = t2-t1
            #e.g.:logits.shape = (20,2)=(batchsize,class), torch.max(logits).shape = (2,20),[0] is value, [1].shape = (10,1),[1] =[0,1,...] 
            #use torch.max on logits without softmax is same as torch.max(softmax(logits),dim=1)[1]
            predict = torch.max(logits,dim=1)[1]
            #caclulate
            batch_right_nums = (predict == y.to(device)).sum().item()
            right_nums+=batch_right_nums
            sample_nums += y.size(0)
            print(f"batch: {i+1}/{len(val_dataloader)}    batch_accuracy: {batch_right_nums/y.size(0):.2f}      batch_time: {t:6f}")
            
        accuracy =right_nums/sample_nums
        
    print(f'Total accuracy: {accuracy:.2f}')
            
    
    
def trans_logits_in_batch_to_result(logits_in_batch:torch.Tensor|np.ndarray)->list:
    """Trans logits of model output to [probabilities, indices]

    Args:
        logits_in_batch (torch.Tensor | np.ndarray): _description_

    Raises:
        TypeError: not Tensor neither ndarray

    Returns:
        list: [probabilities:torch.Tensor, indices:torch.Tensor]
    """
    if isinstance(logits_in_batch,np.ndarray):
        logits_in_batch = torch.from_numpy(logits_in_batch)
    elif isinstance(logits_in_batch,torch.Tensor):
        pass
    else:
        raise TypeError(f'Wrong input type {type(logits_in_batch)}, only support torch tensor and numpy')
    max_result = torch.max(torch.softmax(logits_in_batch,dim=1),dim=1)
    max_probabilities = max_result.values
    max_indices = max_result.indices
    
    return max_probabilities,max_indices




class Onnx_Engine:
    class Standard_Data:
        def __init__(self) -> None:
            self.result = 0
            
        def save_results(self,results:np.ndarray):
            self.result = results
        
        
    def __init__(self,
                 filename:str,
                 if_offline:bool = False,
                 if_onnx:bool = True) -> None:
        """Config here if you wang more

        Args:
            filename (_type_): _description_
        """
        custom_session_options = onnxruntime.SessionOptions()
        
        if if_offline:
            custom_session_options.optimized_model_filepath = filename
            
        custom_session_options.enable_profiling = False          #enable or disable profiling of model
        #custom_session_options.execution_mode =onnxruntime.ExecutionMode.ORT_PARALLEL       #ORT_PARALLEL | ORT_SEQUENTIAL
        if if_onnx:
            
            custom_session_options.add_session_config_entry('session.load_model_format', 'ONNX') # or 'ORT'
        else:
            custom_session_options.add_session_config_entry('session.load_model_format', 'ORT') 
        #custom_session_options.inter_op_num_threads = 2                                     #default is 0
        #custom_session_options.intra_op_num_threads = 2                                     #default is 0
        custom_session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL # DISABLE_ALL |ENABLE_BASIC |ENABLE_EXTENDED |ENABLE_ALL
        custom_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']       # if gpu, cuda first, or will use cpu
        
        
        self.user_data = self.Standard_Data()  
        self.ort_session = onnxruntime.InferenceSession(filename,
                                                        sess_options=custom_session_options,
                                                        providers=custom_providers)

    def standard_callback(results:np.ndarray, user_data:Standard_Data,error:str):
        if error:
            print(error)
        else:
            user_data.save_results(results)
    
    
    @timing(1)            
    def run(self,output_nodes_name_list:list|None,input_nodes_name_to_npvalue:dict)->list:
        """Notice that input value must be numpy value
        Args:
            output_nodes_name_list (list | None): _description_
            input_nodes_name_to_npvalue (dict): _description_

        Returns:
            list: [[node1_output,node2_output,...],reference time]
        """
        return self.ort_session.run(output_nodes_name_list,input_nodes_name_to_npvalue)
    
    
    def eval_run_node0(self,val_data_loader:DataLoader,input_node0_name:str):
        """Warning: only surpport one input and one output; \n
                    only support dynamic onnx model ???    

        Args:
            val_data_loader (DataLoader): _description_
            input_node0_name (str): _description_
        """
           
        right_nums = 0
        sample_nums = 0
        total_time = 0
        for i,sample in enumerate(val_data_loader):
            X,y = sample
            X = X.numpy()
            
            out,once_time = self.run(None,{input_node0_name:X})
            logits = out[0]
            
    
            predict = torch.max(torch.from_numpy(logits),dim=1)[1]
            #caclulate
            batch_right_nums = (predict == y).sum().item()
            right_nums+=batch_right_nums
            sample_nums += y.size(0)
            total_time+=once_time

            print(f"batch: {i+1}/{len(val_data_loader)}    batch_accuracy: {batch_right_nums/y.size(0):.2f}    batch_time: {once_time:6f} ")
          
            
        accuracy =right_nums/sample_nums
        avg_time = total_time/len(val_data_loader)
        
        print(f'Total accuracy: {accuracy:.2f}')
        print(f'Total time: {total_time:2f}')
        print(f'Avg_batch_time: {avg_time}')
            
    @timing(1)
    def run_asyc(self,output_nodes_name_list:list|None,input_nodes_name_to_npvalue:dict):
        """Will process output of model in callback , config callback here

        Args:
            output_nodes_name_list (list | None): _description_
            input_nodes_name_to_npvalue (dict): _description_

        Returns:
            [None,once reference time]
        """
        self.ort_session.run_async(output_nodes_name_list,
                                             input_nodes_name_to_npvalue,
                                             callback=self.standard_callback,
                                             user_data=self.user_data)
        return None


class Onnx_Processer:
    def __init__(self) -> None:
        pass
    @classmethod
    def preprocess(cls,
                   ori_onnx_model_path:str):
        output_onnx_model_path = oso.add_suffix_to_end(ori_onnx_model_path,'_opt')
        
        oq.quant_pre_process(ori_onnx_model_path,
                            output_onnx_model_path,
                            skip_optimization=False,
                            skip_onnx_shape=False,              # works best for non-Transformer
                            skip_symbolic_shape=False,           # works best for Transformer
                            auto_merge=False,
                            verbose=True,
                            save_as_external_data=False,
                            all_tensors_to_one_file=False,
                            external_data_location='./external_data'
                            )
        
        print(f'preprocessed onnx model saved to {output_onnx_model_path}')
        
    @classmethod
    def quantize(cls,
                 preprocess_onnx_model_path:str,
                 calibration_data_loader:DataLoader|None=None,
                 mode:str = 'static',
                 input_nodes_name_list:list = ['input']):
        

        if mode == 'dynamic':
            output_onnx_model_path = oso.add_suffix_to_end(preprocess_onnx_model_path,'_Qdynamic')
            #dynamic run slowest on my CPU
            oq.quantize_dynamic(preprocess_onnx_model_path,
                                output_onnx_model_path,
                                op_types_to_quantize=None, # ['Conov',...]
                                per_channel=False,          # can improve accuracy
                                reduce_range=True,         # use for U8S8 format on non-VNNI machine and for per_channel_True on non-VNNI machine
                                weight_type=oq.QuantType.QUInt8,
                                nodes_to_quantize=None,    # [Node1, Node2 ,...]
                                nodes_to_exclude=None,     # [Node1, Node2, ...]
                                use_external_data_format=False # if save tensors in other files and not in model file
                                )

        elif mode == 'static' :
            output_onnx_model_path = oso.add_suffix_to_end(preprocess_onnx_model_path,'_Qstatic')
            
            if calibration_data_loader is None:
                raise TypeError("calibration data loader can not be None")
            dreader = Dataloader_CalibrationDataReader(calibration_data_loader,input_nodes_name_list)
            
            oq.quantize_static(preprocess_onnx_model_path,
                            output_onnx_model_path,
                            calibration_data_reader=dreader,    # use Trans to get datareader from dataloader
                            quant_format=oq.QuantFormat.QDQ,    # QDQ or QO, QO is larger ?
                            op_types_to_quantize=None,  
                            per_channel=True,                   # accuracy up
                            reduce_range=True,                  # speed up
                            activation_type=oq.QuantType.QUInt8,        #U8S8 run fastest in my CPU
                            weight_type=oq.QuantType.QInt8,
                            nodes_to_quantize=None,
                            nodes_to_exclude=None,
                            use_external_data_format=None,                      
                            calibrate_method=oq.CalibrationMethod.MinMax        # define scaler by max and min of calibration dataset
                            )
        else:
            raise TypeError(f"Wrong mode {mode}, only support dynamic and static")
        
        print(f'quantized_model saved to {output_onnx_model_path}')
        
    @classmethod
    def end_to_end_opt_qua(cls,
                           ori_onnx_model_path:str,
                           calibration_data_loader:DataLoader|None,
                           mode:str = 'static',
                           input_nodes_name_list:list = ['input'],
                           if_save_tmp_refenrece:bool = False):
        name = oso.get_name(ori_onnx_model_path)
        tmp = oso.add_suffix_to_end(ori_onnx_model_path,'_opt')
        cls.preprocess(ori_onnx_model_path)
        cls.quantize(tmp,
                     calibration_data_loader,
                     mode=mode,
                     input_nodes_name_list=input_nodes_name_list)
        
        if not if_save_tmp_refenrece:
            os.remove(tmp)
            print(f'remove {tmp}')
        
        print(f'ALL DONE')

    
    @classmethod
    def trans_onnx_to_ort(cls,onnx_path:str):
        raise TypeError("Not implemented yet")
        output_path = oso.change_ext_only(onnx_path,'.ort')
        output_path = Path(output_path)
        
        convert_onnx_models_to_ort(onnx_path,output_path)

   

        
        



