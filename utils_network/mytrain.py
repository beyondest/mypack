import torch
import sys
sys.path.append('..')
import os_op.os_operation as oso
import os
import time
def train_classification(model:torch.nn.Module,
          train_dataloader,
          val_dataloader,
          device,
          epochs:int,
          criterion,
          optimizer,
          weights_save_path:str|None,
          save_interval:int = 1,
          show_step_interval:int=10,
          show_epoch_interval:int=2
          ):
    '''
    if show_step_interval<0, will not show step
    '''
    t1 = time.perf_counter()
    step_show =True
    epoch_show = True
    if show_step_interval <0:
            step_show = False
    if show_epoch_interval<0:
        epoch_show = False
    val_sample_nums = len(val_dataloader)
    train_sample_nums = len(val_dataloader)
        
        
    save_path = os.path.split(weights_save_path)[0]
        
    print(f'train_sampls:{train_sample_nums}, val_sample_nums:{val_sample_nums}')
    
    best_accuracy = 0
        
    for epoch in range(epochs):
        step = 0
        accuracy_nums = 0
        model.train()
        
        for X,y in train_dataloader:
            model.zero_grad()
            logits = model(X.to(device))
            loss = criterion(logits,y.to(device))
            loss.backward()
            optimizer.step()
            if step_show:
                if step%show_step_interval ==0:
                    print(f'step:{step}/{train_sample_nums}    step_loss:{loss.item():.3f}')
            step +=1
            
            
        model.eval()
        
        with torch.no_grad():
            for X,y in val_dataloader:
                logits = model(X.to(device))
                #e.g.:logits.shape = (20,2)=(batchsize,class), torch.max(logits).shape = (2,20),[0] is value, [1].shape = (10,1),[1] =[0,1,...] 
                #use torch.max on logits without softmax is same as torch.max(softmax(logits),dim=1)[1]
                predict = torch.max(logits,dim=1)[1]
                #caclulate
                accuracy_nums+=(predict == y.to(device)).sum().item()
            accuracy =accuracy_nums/val_sample_nums
            
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
    t2 = time.perf_counter()
    t = t2-t1
    print(f"Training over,Spend time:{t:.2f}s")
    print(f"Best accuracy : {best_accuracy} ")