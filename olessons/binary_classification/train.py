import numpy as np
import torch
import sklearn.datasets
import sys
sys.path.append('../../utils_network/')
from utils_network.data import *
from utils_network.mymodel import *
from params import *
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import yaml
if __name__=='__main__':

    feature_nums,sample_nums=Data.get_feature_sample_nums_from_yaml(yaml_save_path)
    
    if generate_mode==0:
        model=simple_2classification(feature_nums=feature_nums,feature_expand=True,correlation=correlation)
    else:
        model=simple_2classification(feature_nums=feature_nums)
    #set optimizer: SGD = Stochastic Gradient Descent, choose mini-batch of X randomly to caculate gradient
    optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)
    #choose loss function, Binary Cross Entropy Loss
    criterion=torch.nn.BCELoss()
    

    dataset=np.load(data_save_path)
    X=dataset['X']
    y=dataset['y']
    
    if model.feature_expand:
        X=Data.feature_expand(X,feature_nums,correlation)
    
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)
    X_train,X_test,y_train,y_test=map(lambda data: torch.from_numpy(data).to(dtype=torch.float32),[X_train,X_test,y_train,y_test])
    y_train,y_test=map(lambda data: data.reshape(-1,1),[y_train,y_test])
    dataset=TensorDataset(X_train,y_train)
    
    
    #if shuffle, then loss will go down apparently, but better to set True
    dataloader=DataLoader(dataset,batch_size,shuffle=True)
    
    for epoch in range(epochs):
        for batch_data in dataloader:
            sample_input,sample_labels=batch_data
            optimizer.zero_grad()
            
            outputs =model(sample_input)
            
            loss = criterion(outputs, sample_labels)
            
            loss.backward()
            #update optimizer
            optimizer.step()
            
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')

    
    print(model.state_dict())
    print(f'model save to {weights_save_path}')
    torch.save(model.state_dict(),weights_save_path)
    
    
    
with torch.no_grad():
    model.eval()
    predictions=model(X_test)

binary_predictions=(predictions>=confidence).float()


accuracy=accuracy_score(y_test.numpy(),binary_predictions.numpy())
print(f'Accuracy: {accuracy}')




