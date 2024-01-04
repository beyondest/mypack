import torch
import sys
sys.path.append('..')
import os_op.os_operation as oso
import os
import time
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms,datasets
import cv2
from utils_network.data import *

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
                           img_path:str,
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
        img_path (str): _description_
        weights_path (str): _description_
        class_yaml_path (str): _description_
        fmt (str, optional): _description_. Defaults to 'jpg'.
        if_cvt_rgb (bool, optional): _description_. Defaults to True.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    idx_to_class = Data.get_file_info_from_yaml(class_yaml_path)
    model.load_state_dict(torch.load(weights_path,map_location=device))
    model.eval()
    suffix = os.path.splitext(img_path)[-1]
    if suffix != '':
        mode = 'single_img'
    else:
        mode = 'multi_img'
    
    def single_predic(img_path):
        img_ori = cv2.imread(img_path)
        if custom_trans_cv is not None:
            img = custom_trans_cv(img_ori)
        else:
            img = img_ori   
            
        if if_show:
            imo.cvshow(img,f'{img_path}')
        input_tensor = trans(img).unsqueeze(0).to(device)

        
        with torch.no_grad():
            logits = model(input_tensor)
            
            probablility_tensor = torch.nn.functional.softmax(logits,dim=1)[0]
            probablility = torch.max(probablility_tensor).item()
            predict_class_id = torch.argmax(probablility_tensor).item()
            predict_class = idx_to_class[predict_class_id]
            if if_print:
                print(f'{img_path} is {predict_class}, with probablity {probablility:.2f}')

        if if_draw_and_show_result:
            imo.add_text(img_ori,f'class:{predict_class} | probability: {probablility}',0,color=(0,255,0))
            imo.cvshow(img_ori,'result')
            
    if mode == 'single_img':
        single_predic(img_path)
        

        
    else:
        oso.traverse(img_path,None,deal_func=single_predic,fmt=fmt)
        
        
        
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
            logits = model(X.to(device))
            
            #e.g.:logits.shape = (20,2)=(batchsize,class), torch.max(logits).shape = (2,20),[0] is value, [1].shape = (10,1),[1] =[0,1,...] 
            #use torch.max on logits without softmax is same as torch.max(softmax(logits),dim=1)[1]
            predict = torch.max(logits,dim=1)[1]
            #caclulate
            batch_right_nums = (predict == y.to(device)).sum().item()
            right_nums+=batch_right_nums
            sample_nums += y.size(0)
            print(f"batch: {i+1}/{len(val_dataloader)}    batch_accuracy: {batch_right_nums/y.size(0):.2f}")
        accuracy =right_nums/sample_nums
    print(f'Total accuracy: {accuracy:.2f}')
            
    
    


    
    
        

