#only generate mode 0 can visualize

import numpy as np

def func0(x):
        return np.power(x,2)
def func1(x):
    return np.power(x,3)

#path
data_save_path='../../datasets/dataset1.npz'
weights_save_path='./weights1.pth'
data_fromnet_path='/mnt/d/datasets/heart_disease/heart.csv'  #change to your own path, notice the dataset must be binary classification datastet
yaml_save_path="./train_params.yaml"
#generate_mode:
'''
0: draw 2 closed shape to generate points inside them,use these points as dataset
1: use dataset download from internet
2: use dataset from sklearn
'''


generate_mode=0

#default:1000,6
#only work for mode 0 and mode 2, mode 1 has its own nums
sample_nums=1000
feature_nums=6


correlation=[[[0],[0],func1],[[1],[1],func1],[[0],[2],func0],[[1],[3],func0],[[0],[4]],[[1],[5]]]


learning_rate=0.01
batch_size=200
epochs=5000
confidence = 0.5

