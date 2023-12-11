import numpy as np
import torch
import sklearn.datasets
import sys
sys.path.append('../../utils_network/')
from utils_network.data import *
from utils_network.mymodel import *
from params import *
from sklearn.metrics import accuracy_score


feature_nums,sample_nums=Data.get_feature_sample_nums_from_yaml(yaml_save_path)
if generate_mode == 0:
    
    model=simple_2classification(feature_nums,True,correlation=correlation)

    model.load_state_dict(torch.load(weights_save_path))

    print(model.state_dict())

    hp_pts=Data.get_hyperplane_pts(model,pt_nums=200)
    d=Data.binary_classification_dataset(sample_nums=sample_nums,
                                        xlim=(-5,5),
                                        ylim=(-5,5),
                                        path=data_save_path)
    d.show2d(hp_points=hp_pts)

elif generate_mode ==1:
    model = simple_2classification(feature_nums=feature_nums)
    model.load_state_dict(torch.load(weights_save_path))
    print(model.state_dict())
    model.eval()
    
        
    ori_data=pd.read_csv(data_fromnet_path)
    X,y=Data.data_read_classification(data_fromnet_path)
        
        
    X=torch.from_numpy(X).to(dtype=torch.float32)
    with torch.no_grad():
        predictions = model(X)
        predictions =(predictions>=confidence).float()
    ori_data["predictions"]= predictions.numpy()
    accuracy = accuracy_score(y,predictions.numpy())
        
    pd.set_option('display.max_rows',None)
    print(ori_data)
    print(f'accuracy is : {accuracy}')



elif generate_mode ==2:
    model = simple_2classification(feature_nums=feature_nums)
    model.load_state_dict(torch.load(weights_save_path))
    print(model.state_dict())
    model.eval()
    
    ori_data= np.load(data_save_path)
        
        
    X=ori_data['X']
    y=ori_data['y']
    X=torch.from_numpy(X).to(dtype=torch.float32)
    with torch.no_grad():
        predictions = model(X)
        predictions = (predictions >= confidence).float()
        
    accuracy = accuracy_score(y,predictions.numpy())
    print(f'accuracy is {accuracy}')


