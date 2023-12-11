import numpy as np
import torch
import sklearn.datasets as skd
import sys
sys.path.append('../../..')
sys.path.append('../../utils_network/')
from utils_network.data import *
from utils_network.mymodel import *
from params import *
import pandas as pd
from params import *

if __name__=="__main__":

    

    if generate_mode==0:
        
        d=Data.binary_classification_dataset(sample_nums=sample_nums,
                            class_nums=2,
                            xlim=(-5,5),
                            ylim=(-5,5)
                            )
        d.generate_data()
        d.save(data_save_path=data_save_path)
        print(f'feature_nums: {feature_nums}')
        print(f'sample_nums: {sample_nums}')
        
        
    elif generate_mode==1:
        
        X,y=Data.data_read_classification(data_fromnet_path)
        
        
        sample_nums=X.shape[0]
        feature_nums=X.shape[1]
        class_nums=y.max()+1
        assert class_nums==2,'this net can only handle binary classification task, class num has to be 2'
        np.savez(data_save_path,X=X,y=y)
        print(f'data saved to {data_save_path}')
        print(f'feature_nums: {feature_nums}')
        print(f'sample_nums: {sample_nums}')
        
        
        
    elif generate_mode==2:
        X,y=skd.make_classification(
            n_samples=sample_nums,
            n_features=feature_nums,
            n_informative=1,
            n_redundant=0,
            n_repeated=0,
            n_classes=2,
            n_clusters_per_class=1
        )
        
        np.savez(data_save_path,X=X,y=y)
        print(f'data saved to {data_save_path}')
        print(f'feature_nums: {feature_nums}')
        print(f'sample_nums: {sample_nums}')
        
    else:
        raise TypeError(f"generate mode {generate_mode} is not supported")
    
    
    Data.save_feature_sample_nums_to_yaml(feature_nums,sample_nums,yaml_save_path)
    
        
        
    