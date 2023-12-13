import torch
import sys
sys.path.append('..')
import os_op.os_operation as oso
import os
import time
from torch.utils.tensorboard import SummaryWriter


def train_classification(model:torch.nn.Module,
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
    save_path = os.path.split(weights_save_path)[0]
    
    
    best_accuracy = 0
    epoch_loss = 0
      
    for epoch in range(epochs):
        step = 0
        step_loss = 0
        
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
                    print(f'step:{step}/{train_step_nums}   present_step_loss:{loss.item():.3f}')
            if log_step_write:
                if step%log_step_interval == 0:
                    writer.add_scalar('avg_step_loss',step_loss/(step+1),epoch * train_step_nums+step)

            
        epoch_loss+=step_loss 
        if log_epoch_write:
            if epoch%log_epoch_interval == 0:
                writer.add_scalar('avg_epoch_loss',epoch_loss/(epoch+1),epoch)
         
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
        
        
        writer.add_scalar('epoch_accuracy',accuracy,epoch)
        
        if accuracy>best_accuracy:
            best_accuracy = accuracy
        
        
        if epoch_show:   
            if epoch%show_epoch_interval== 0:
                oso.get_current_datetime(True)
                print(f"epoch:{epoch+1}/{epochs}    epoch_loss:{loss.item():.3f}         accuracy:{accuracy:.2f}         best_accuracy:{best_accuracy}")
        
        
        if weights_save_path is not None:
        
            if epoch%save_interval == 0:
                name = f'weights.{accuracy}.{epoch}.pth'
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
    