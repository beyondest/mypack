import sys
sys.path.append('../../')
from params import *
from utils_network.data import *
from torchvision import transforms,datasets
from torch.utils.data import Dataset,DataLoader
from utils_network.tf_data import *


train_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(0.5),
    transforms.Normalize(mean,std)
])
val_trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean,std)
])

train_dataset  =  datasets.ImageFolder(train_root_path,train_trans)
val_dataset = datasets.ImageFolder(val_root_path,val_trans)


Data.save_dict_info_to_yaml(val_dataset.class_to_idx,class_yaml_path)
tfrecord.save_to_tfrecords(val_dataset,val_tfrecords_path,save_y_type=int)

#Data.save_dataset_to_pkl(train_dataset,train_pkl_path)
#Data.save_dataset_to_pkl(val_dataset,val_pkl_path)
#Data.save_dataset_to_pkl(train_dataset,train_pkl_path)
#tfrecord.save_to_tfrecords(val_dataset,test_tfrecords_path)








