
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

train_path,val_path,weights_save_path = Data.get_path_info_from_yaml(yaml_path)


hdataset = datasets.ImageFolder
val_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224))
])

train_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(flip_probability),
    transforms.Normalize(mean,std)
])
val_dataset = dataset_pkl(val_path)
train_dataset = dataset_pkl(train_path)


val_dataloader = DataLoader(val_dataset,
                        batchsize,
                        shuffle=False,
                        num_workers=1)

train_dataloader = DataLoader(train_dataset,
                              batchsize,
                              shuffle=True,
                              num_workers=2)


model = mobilenet_v2(num_classes=2)
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)


train_classification(model,
                     train_dataloader,
                     val_dataloader,
                     device,
                     epochs,
                     criterion,
                     optimizer,
                     weights_save_path,
                     save_interval=3,
                     show_step_interval=4,
                     show_epoch_interval=1)

















            
            
    
    
    






