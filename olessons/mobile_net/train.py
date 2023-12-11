
from torchvision import transforms, datasets
import torchvision.models.mobilenet
import sys
sys.path.append('../../')
from params import *
from utils_network.data import *
from torchvision import transforms,datasets
from torch.utils.data import Dataset,DataLoader
import os_op.os_operation as oso
from utils_network.mytrain import *
from utils_network.mymodel import *



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device:{device}')



hdataset = datasets.ImageFolder
val_dataset = Data.get_dataset_from_pkl(hdataset,val_pkl_path)
        
val_dataloader = DataLoader(val_dataset,
                        batchsize,
                        shuffle=True,
                        num_workers=1)

model = mobilenet_v2(num_classes=2)
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)


train_classification(model,
                     val_dataloader,
                     val_dataloader,
                     device,
                     epochs,
                     criterion,
                     optimizer,
                     weights_savepath,
                     save_interval=3,
                     show_step_interval=4,
                     show_epoch_interval=1)











            
            
    
    
    






