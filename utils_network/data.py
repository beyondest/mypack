import numpy as np
import random 
import time
import matplotlib.pyplot as plt
import math
import sys
sys.path.append('..')
from utils_network.mymath import *
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from shapely.geometry import Point, Polygon
import torch
from utils_network.mymodel import *
import os_op.os_operation as oso
import os
import yaml
from torch.utils.data import Dataset,DataLoader
import gzip
import pickle
from torch.utils.data import TensorDataset
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import fcntl
import threading
import h5py

def tp(*args):
    for i in args:
        print(type(i))







class Data:
    def __init__(self,file_path:str|None=None,seed:int=10) -> None:
        random.seed(seed)        
        np.random.seed(seed)
    
      
    class plt_figure:
        def __init__(self,figsize=(10,10)) -> None:
            self.figure=plt.figure(figsize=figsize)
            self.ax=[]
            self.ax_nums=0
            self.figure.canvas.mpl_connect("key_press_event",self.key_close)
            
        def key_close(self,event):
            if event.key=="escape":
                plt.close(event.canvas.figure)
            
        def plt_point(self,
                        draw_method:str='scatter',
                        ax_index:int|None=None,
                        x:np.ndarray|tuple=(-5,5),
                        y:np.ndarray|None=None,
                        z:np.ndarray|None=None,
                        projection:str|None=None,
                        position:int=111,
                        func=None,
                        xlim:tuple|None=None,
                        ylim:tuple|None=None,
                        zlim:tuple|None=None,
                        show_now:bool=False,
                        x_nums:int=1000,
                        color:str='b')->int:
            """plots or scatter after figure init. uf ax_index is specified, then draw on specified ax, else generate new ax\n
            Returns:
                int: ax_index
            Args:
                draw_method (str, optional): _description_. Defaults to 'scatter'.
                ax_index (int | None, optional): _description_. Defaults to None.
                x (np.ndarray | tuple, optional): _description_. Defaults to (-10,10).
                y (np.ndarray | None, optional): _description_. Defaults to None.
                z (np.ndarray | None, optional): _description_. Defaults to None.
                projection (str | None, optional): _description_. Defaults to None.
                position (int, optional): _description_. Defaults to 111.
                func (_type_, optional): _description_. Defaults to None.
                xlim (tuple | None, optional): _description_. Defaults to None.
                ylim (tuple | None, optional): _description_. Defaults to None.
                zlim (tuple | None, optional): _description_. Defaults to None.
                show_now (bool, optional): _description_. Defaults to False.
                x_nums (int, optional): if x is default, generate linspace x in x_nums. Defaults to 1000.
                color (str, optional): _description_. Defaults to 'b'.

            
            """
            #geneate axis
            if ax_index is None:
                if projection is not None:
                    self.ax.append(self.figure.add_subplot(position,projection=projection))
                    
                else:
                    self.ax.append(self.figure.add_subplot(position))
                ax=self.ax[self.ax_nums]
                self.ax_nums+=1
                out=self.ax_nums-1
                
            else:
                
                ax=self.ax[ax_index]
                out=ax_index
            #generate points
            if isinstance(x,tuple):
                x=np.linspace(x[0],x[1],x_nums)
            if func is not None:
                y=func(x)
            
            #draw 
            if draw_method=='scatter':
                if z is not None :
                    ax.scatter(x,y,z,c=color)
                else:
                    ax.scatter(x,y,c=color)
            else:
                if z is not None :
                    ax.plot(x,y,z,c=color)
                else:
                    ax.plot(x,y,c=color)
            #custom settings
            if ax_index==None:
                if xlim is not None:
                    ax.set_xlim(xlim)
                if ylim is not None:
                    ax.set_ylim(ylim)
                if zlim is not None and projection =='3d':
                    ax.set_ylim(zlim)
                
                
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                if projection=='3d':
                    ax.set_zlabel("z")
            
            
            if show_now:
                
                plt.show()
            return out
    
    class random_point:
        def __init__(self,
                    dim:int,
                    sample_nums:int,
                    scope:tuple,
                    correlation:list|None=None,
                    noise:list|None=None,
                    )->np.ndarray:
            """generate points uniform destributed in independent dims,dependent dim generate by func, return points in np.ndarray
            NOTICE: random_points.all_points has (samples*features) shape, same as most of dataset!!!
            Args:
                dim (int): the row of out_mat, the dimension of eigens\n
                sample_nums (int): the col of out_mat, the numbers of samples\n
                scope (tuple): the scope of uniform generate\n
                \n
                correlation (list | None):[[[dim1,dim2...input_dim],[dim0,dim3...output_dim],func]]\n
                    NOTICE:func must input (samples,dims) ndarray and output ndarray\n
                    NOTICE: list contains list contains list at least\n
                    NOTICE: if input_dim=output_dim, then will use func to generate axis
                    e.g.:\n
                    [[0,1],[2,3],myaddfunc]\n
                \n
                noise (list | None): [[[dim0,dim1...],"type",(arg1,arg2...)],[[dim3,dim5...],"type",(arg1,arg2...)]]\n
                    NOTICE: custom_func is INVERSE of distribution function of X,input and output must be np.ndarray\n
                    NOTICE: list contains list contains list at least\n
                    NOTICE: each outlier add to origin, not replace,so that noise can add too!!!\n
                    e.g.\n
                    [[[0,1],"normal",(0=u,1=thegma^2)] , [[0],"uniform",(-1=a,1=b)] ]\n
                    [[[0,1,2],"pulse",(0.1=percent of outliers,(10,20)=abs_scope,True = make negative_pulse)] ,[[0,1],"custom",func]]\n
            """
            self.scope=scope
            self.sample_nums=sample_nums
            self.count=0
            if correlation==None:
                self.all_points=np.random.uniform(*scope,(sample_nums,dim))
            else:
                
                out=np.random.uniform(*scope,(sample_nums,dim))
                for each in correlation:
                        
                    input_dim_array=np.array(each[0])
                    output_dim_array=np.array(each[1])
                    #if inputdim is same as outputdim, then make this dim as axis
                    if input_dim_array.all()==output_dim_array.all() and len(input_dim_array)==1:
                        out[:,output_dim_array]=self.make_axis()
                    else:
                        out[:,output_dim_array]=each[2](out[:,input_dim_array])
                if noise is not None:
                    for each in noise:
                        dim_array=np.array(each[0])
                        if each[1]=="normal":
                            u=each[2][0]
                            thegma=np.sqrt(each[2][1])
                            out[:,dim_array]=out[:,dim_array]+(u+thegma*np.random.randn(sample_nums,len(dim_array)))
                        elif each[1]=="uniform":
                            low=each[2][0]
                            high=each[2][1]
                            out[:,dim_array]=out[:,dim_array]+np.random.uniform(low,high,(sample_nums,len(dim_array)))
                        elif each[1]=="pulse":
                            outlier_nums=round(each[2][0]*sample_nums)
                            outlier_scope=each[2][1]
                            outlier=np.random.uniform(outlier_scope[0],outlier_scope[1],outlier_nums)
                            if each[2][2]:
                                negative_nums=round(np.random.rand()*outlier_nums)
                                negative_indices=np.random.choice(len(outlier),negative_nums,replace=False)
                                outlier[negative_indices]=-outlier[negative_indices]
                            
                            
                            
                            flatten_matrix=out[:,dim_array].flatten()
                            outlier_indices=np.random.choice(len(flatten_matrix),outlier_nums,replace=False)
                            flatten_matrix[outlier_indices]=flatten_matrix[outlier_indices]+outlier
                            out[:,dim_array]=flatten_matrix.reshape(out[:,dim_array].shape)
                            
                        elif each[1]=="custom":
                            inverse_distribution=each[2]
                            out[:,dim_array]=out[:,dim_array]+inverse_distribution(np.random.rand(sample_nums,len(dim_array)))
                            
                self.all_points=out
                
                
        
            
        def scatter2d(self,
                      position:int=111,
                      xlim:tuple|None=None,
                      ylim:tuple|None=None,
                      figsize=(10,10),
                      show_now:bool=True):
            '''
            show points 2d quickly
            
            '''
            f=Data.plt_figure(figsize=figsize)
            f.plt_point(x=self.all_points[:,0],
                        y=self.all_points[:,1],
                        position=position,
                        xlim=xlim,
                        ylim=ylim,
                        show_now=show_now)
            
        def scatter3d(self,
                      position:int=111,
                      xlim:tuple|None=None,
                      ylim:tuple|None=None,
                      zlim:tuple|None=None,
                      figsize=(10,10)):
            '''
            show all_points 3d quickly
            '''
            f=Data.plt_figure(figsize=figsize)
            f.plt_point(x=self.all_points[:,0],
                        y=self.all_points[:,1],
                        z=self.all_points[:,2],
                        projection='3d',
                        show_now=True,
                        xlim=xlim,
                        ylim=ylim,
                        zlim=zlim,
                        position=position)
            
            
        def append(self,new_point:np.ndarray):
            '''
            use c_ to add new_point, notice shape must same
            '''
            
            self.all_points=np.r_[self.all_points,new_point]
            
            
        def make_axis(self):
            return np.linspace(self.scope[0],self.scope[1],self.sample_nums).reshape(-1,1)
    
    
    class binary_classification_dataset:
        def __init__(self,
                     mode:int=0,
                     sample_nums:int|list=500,
                     class_nums:int=2,
                     xlim:tuple=(0,1),
                     ylim:tuple=(0,1),
                     draw_size:tuple=(15,12),
                     path:str|None=None) -> None:
            """generate or read binary classification dataset 

            Args:
                mode (int): 
                    0: generate data by draw curve or by same dataset by path; can only have 2 features
                    1: generate data by sklearn
                    2: read data by path, more than 2 features
                sample_nums (int | list, optional): _description_. Defaults to 500.
                class_nums (int, optional): _description_. Defaults to 2.
                xlim (tuple, optional): _description_. Defaults to (0,1).
                ylim (tuple, optional): _description_. Defaults to (0,1).
                draw_size (tuple, optional): _description_. Defaults to (15,12).
                path (str | None, optional): _description_. Defaults to None.
            """
            self.colors=['r','g','b','c','m','y','k','w']
            self.labels=[i for i in range(class_nums)]
            self.class_nums=class_nums
            self.xlim=xlim
            self.ylim=ylim
            self.draw_size=draw_size
            self.sample_nums=sample_nums
            self.points_curve_list=None
            self.points_inside_list=None
            if path is not None:
                
                self.points_inside_list=[]
                for i in range(self.class_nums):
                    self.points_inside_list.append([])
                dataset=np.load(path)
                X=dataset['X'].reshape(-1,self.class_nums)
                y=dataset['y'].flatten()
                for i in range(len(y)):
                    if y[i]==0:
                        self.points_inside_list[0].append(X[i])
                    else:
                        self.points_inside_list[1].append(X[i])
                for i in range(self.class_nums):
                    self.points_inside_list[i]=np.array(self.points_inside_list[i]).reshape(-1,self.class_nums)

        
        
        
        def _generate_points_inside_curve(self,vertices_list:list|np.ndarray, num_points)->list:
            '''
            generate points inside curve by uniform random between (x_min,x_max) and (y_min,y_max)\n
            vertices:contours points, shape like (n,2)\n
            return points_inside_list
            '''
            points_inside_list=[]
            if isinstance(vertices_list,np.ndarray):
                vertices_list=[vertices_list]
            for index,i in enumerate(vertices_list):
               
                points_inside_list.append([])
                polygon = Polygon(i)

                # Get the bounding box of the polygon
                min_x, min_y, max_x, max_y = polygon.bounds

                while len(points_inside_list[index]) < num_points:
                    # Generate random points inside the bounding box
                    x = np.random.uniform(min_x, max_x)
                    y = np.random.uniform(min_y, max_y)

                    point = Point(x, y)

                    # Check if the point is inside the polygon
                    if point.within(polygon):
                        points_inside_list[index].append((x, y))
                
                points_inside_list[index]=np.array(points_inside_list[index])
                
            return points_inside_list
        
        def generate_data(self):
            '''
            return X,y\n
            X.shape=(-1,2)\n
            y.shape=(-1,1)
            '''
            capture=Mouse_trajectory_capture(self.class_nums,self.xlim,self.ylim,self.draw_size)
            points_curve_list=capture.start_draw()
            points_inside_list=self._generate_points_inside_curve(points_curve_list,self.sample_nums)
            self.points_curve_list=points_curve_list
            self.points_inside_list=points_inside_list
            label_list=self._make_labels()
            X=np.array(points_inside_list,dtype=np.float32).reshape(-1,2)
            y=np.array(label_list,dtype=np.float32).reshape(-1,1)
            self.X=X
            self.y=y
            return X,y
        
        def feature_expand(self,dst_feature_nums:int,correlation:list):
            X=Data.feature_expand(self.X,dst_feature_nums=dst_feature_nums,correlation=correlation)
            self.X=X
            return X
        
        def show2d(self,hp_points:np.ndarray|None=None):
            """show hyperplane together with datasets points

            Args:
                hp_points (np.ndarray | None, optional): _description_. Defaults to None.
            """
            
            f_obj=Data.plt_figure()
            ax=f_obj.figure.add_subplot()
            fig=f_obj.figure
            
            if hp_points is not None:
                ax.scatter(hp_points[:,0],hp_points[:,1],c=self.colors[self.class_nums],label='decision boundary')
            for index in range(self.class_nums):
                if self.points_curve_list is not None:
                    
                    ax.plot(self.points_curve_list[index][:,0],self.points_curve_list[index][:,1],c=self.colors[index],linewidth=2)
                if self.points_inside_list is not None:
                    ax.scatter(self.points_inside_list[index][:,0],self.points_inside_list[index][:,1],c=self.colors[index],label=self.labels[index])
            plt.legend()
            plt.show()
        
        
        def _make_labels(self)->list:
            """generate label_array correspond to datapoints

            Returns:
                list: _description_
            """
            
            label_list=[]
            
            for i in range(self.class_nums):
                
                k=np.zeros((self.sample_nums,1),dtype=np.float32)
                k.fill(self.labels[i])
                label_list.append(k)
            return label_list
        
        
        def save(self,name:None|str='test',data_save_path:str="./test.npz"):
            if os.path.splitext(data_save_path)[1]!='.npz':
                data_save_path=os.path.join(data_save_path,name+'.npz')
            print(f'data saved to {data_save_path}')
            np.savez(data_save_path,X=self.X,y=self.y)
    
        
    @classmethod
    def get_hyperplane_pts( cls,
                            model:simple_2classification,
                            xlim:tuple=(-5,5),
                            ylim:tuple=(-5,5),
                            pt_nums:int=1000,
                            threshold:float=0.1)->np.ndarray:
        """ fit hyperplane points via simple_2classification model you input ,this model should has been trained already
            ,return np.ndarray ,shape is (pt_nums,2)

        Args:
            model (simple_2classification): _description_
            xlim (tuple, optional): _description_. Defaults to (-5,5).
            ylim (tuple, optional): _description_. Defaults to (-5,5).
            pt_nums (int, optional): _description_. Defaults to 1000.
            threshold (float, optional): distance the point generate randomly with hyperplane. Defaults to 0.1.

        Returns:
            np.ndarray: _description_
        """
        pts_list=[]
        while len(pts_list)<pt_nums:
            x=np.random.uniform(xlim[0],xlim[1])
            y=np.random.uniform(ylim[0],ylim[1])
            ori_pt=np.array([x,y]).reshape(1,-1)
            
            if model.feature_expand:
                data_input=Data.feature_expand(ori_pt,model.fc.in_features,model.correlation)
            else:
                data_input=ori_pt
            data_input=torch.from_numpy(data_input)
            data_input=data_input.to(torch.float32).reshape(1,-1)
            
            out=model.fc(data_input)
            out=out.item()
            if abs(out)<threshold:
                pts_list.append(ori_pt)
                
        return np.array(pts_list).reshape(-1,2)
    
    @classmethod
    def feature_expand(cls,
                       ori_features_data:np.ndarray,
                       dst_feature_nums:int,
                       correlation:list|None=None
                       )->np.ndarray:
        '''
        input data has shape like (samples,features)\n
        correlation: [[[input_dim...],[output_dim...],func0],[input_dim...],[output_dim...],func1]\n
        func must input (samples,dims) np.ndarray and output np.ndarray
        '''
        if len(ori_features_data.shape)>2:
            raise TypeError('ori_fieatures_data has wrony shape')
        if len(ori_features_data)==1:
            ori_features_data=ori_features_data.reshape(1,-1)
        sameples,features=ori_features_data.shape
        out=np.zeros((sameples,dst_feature_nums),dtype=np.float32)
        out_judgement=[i for i in range(dst_feature_nums)]
        
        for each in correlation:
                
            input_dim_array=np.array(each[0])
            output_dim_array=np.array(each[1])
            for i in each[1]:
                if i in out_judgement:
                    out_judgement.remove(i)
                else:
                    raise TypeError("correlation wrong, each outdim can only be specified once")
            #if inputdim is same as outputdim, then make this dim as axis
            if len(each)<3:
                out[:,output_dim_array]=ori_features_data[:,input_dim_array]
            else:
                out[:,output_dim_array]=each[2](ori_features_data[:,input_dim_array])
                
        if len(out_judgement)>0:
            raise TypeError(f"correlation wrong, each outdim must be specifide once,{out_judgement} not specified yet")
        return out
            
    
        
    
    @classmethod
    def show_nplike_info(cls,
                     for_show:np.ndarray|list|tuple,
                     precision:int=2,
                     threshold:int=1000,
                     edgeitems:int=3,
                     linewidth:int=75,
                     suppress:bool=False):
        '''
        precision: how many bit after point for float nums\n
        threshold: how many nums to show a summary instead of show whole array\n
        edgeitems: how many nums to show in a summary at begin of a row\n
        linewidth: how many nums to show in a row\n
        suppress: if True , always show float nums NOT in scientifc notation
        '''
        np.set_printoptions(precision=precision,
                            threshold=threshold,
                            edgeitems=edgeitems,
                            linewidth=linewidth,
                            suppress=suppress)
        torch.set_printoptions(precision=precision,
                               threshold=threshold,
                               edgeitems=edgeitems,
                               linewidth=linewidth)
        
        if isinstance(for_show,np.ndarray): 
            print('shape:',for_show.shape)
            print('dtype:',for_show.dtype)
            print('max:',for_show.max())
            print('min:',for_show.min())
            print(for_show)
        elif isinstance(for_show,list):
            count=0
            for i in for_show:
                if not isinstance(i,torch.Tensor) and not isinstance(i,np.ndarray) and not isinstance(i,int):
                    raise TypeError(f'unsupported input type {type(i)},only support np.ndarray and torch.Tensor')
                if isinstance(i,int):
                    print(f'*********{count}*********')
                    print('type:int,value:',i)
                else:
                    
                    print(f'*********{count}*********')
                    print('shape:',i.shape)
                    print('dtype:',i.dtype)
                    print('max:',i.max())
                    print('min:',i.min())
                    print('content:',i)
                count+=1
            
                
                

    def make_taylor_basis(self,column:np.ndarray,order:int=3)->np.ndarray:
        '''
        use column vector x to generate [1 x x^2 x^3 ... x^order] matrix\n
        return matrix  
        
        '''
        column=np.copy(column)
        out=np.ones_like(column)
        for power_index in range(1,order+1):
            out=np.c_[out,np.power(column,power_index)]   
        return out

    def make_fourier_basis(self,column:np.ndarray,order:int=100):
        '''
        use column vector x to generate [1 cosx sinx cos2x sin2x ... cos(order*x) sin(order*x)] matrix\n
        return matrix  
        
        '''
        column=np.copy(column)
        out=np.ones_like(column)    #simulate cos0x
        for multiple_index in range(1,order):
            out=np.c_[out,np.cos(multiple_index*column)]
            out=np.c_[out,np.sin(multiple_index*column)]
        return out
    
    @classmethod
    def data_read_classification(cls,csv_path:str,pre_process:bool=True)->tuple:
        """return X,y ;X: samples:np.ndarray, y labels:np.ndarray

        Args:
            csv_path (str): _description_

        Returns:
            tuple: _description_
        """
           
        ori_data=pd.read_csv(csv_path)
        X=ori_data.iloc[:,:-1]
        y=ori_data.iloc[:,-1]
        
        label_encoder=LabelEncoder()
        scaler=StandardScaler()
        y=label_encoder.fit_transform(y)
        
        if pre_process:
            X=scaler.fit_transform(X)
        else:
            X=X.to_numpy(dtype=np.float32)

        
        
        return X,y
    
            
    @classmethod
    def save_feature_sample_nums_to_yaml(cls,feature_nums:int,sample_nums:int,save_path:str="./train_param.yaml",open_mode:str="w"):
        """save feature nums and sample nums to yaml file after getting dataset

        Args:
            feature_nums (int): _description_
            sample_nums (int): _description_
            save_path (str, optional): _description_. Defaults to "./train_param.yaml".
            open_mode (str, optional): _description_. Defaults to "w".
        """
        forsave={
            "feature_nums": feature_nums,
            "sample_nums":  sample_nums,
            "saving_time": time.asctime()
        }      
        with open(save_path, open_mode) as file:
            yaml.dump(forsave,file)
            print(f"feature_nums and sample_nums saved to {save_path}")
   
    @classmethod
    def get_feature_sample_nums_from_yaml(cls,
                                          yaml_path:str,
                                          open_mode:str="r")->tuple:
        """get feature nums and sample nums from yaml file 

        Args:
            yaml_path (str): _description_
            open_mode (str, optional): _description_. Defaults to "r".

        Returns:
            tuple: _description_
        """
        with open(yaml_path,open_mode) as file:
            print(yaml_path)
            print(os.getcwd())
            data = yaml.safe_load(file)

            
            
            return data["feature_nums"],data["sample_nums"]
    
    
    @classmethod 
    def get_file_info_from_yaml(cls,yaml_path:str,open_mode:str = 'r')->dict:
        
        with open(yaml_path,open_mode) as file:
            config = yaml.safe_load(file)
            
        return config
    
    
    @classmethod
    def save_dict_info_to_yaml(cls,dict_info:dict,yaml_path:str,open_mode:str = 'w'):
        
        with open(yaml_path,open_mode) as file:
            yaml.dump(dict_info,file,default_flow_style=False)
        print(f'dict_info saved to yamlfile in {yaml_path}')
    
    @classmethod
    def save_dataset_to_pkl(cls,
                            dataset:Dataset,
                            pkl_path:str,
                            if_multi:bool = False,
                            max_workers:int = 5):
        """Use 'ab' to write pkl.gz file, first data is length of dataset

        Args:
            dataset (Dataset): _description_
            pkl_path (str): _description_
            if_multi (bool, optional): _description_. Defaults to True.
        """
        print(f'ALL {len(dataset)} samples to save')

        global_lock = threading.Lock()
        proc_bar = tqdm(len(dataset),desc="Saving to pkl:")
        
        with gzip.open(pkl_path,'ab') as f:
            pickle.dump(len(dataset),f)
            
        def save_sample(sample, file_path):
            while global_lock.locked():
                time.sleep(0.02)
                continue
            global_lock.acquire()
            
            x, y = sample
            
            proc_bar.update(1)
            with gzip.open(file_path, 'ab') as f:
                pickle.dump((np.asanyarray(x),np.asanyarray(y)), f)
            global_lock.release() 
                
        if if_multi:
               
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:  
                futures = [executor.submit(save_sample, sample, pkl_path) for sample in dataset]

                concurrent.futures.wait(futures)
        else:
            for sample in dataset:
                save_sample(sample,pkl_path)
        
        proc_bar.close()
        print(f'dataset saved to {pkl_path}')
        print('Notice: when you need to open it, please include your dataset defination in open code')

    
    @classmethod
    def save_dataset_to_pkl2(cls,
                             dataset,
                             pkl_path,
                             max_workers):
        """Dont use this now

        Args:
            dataset (_type_): _description_
            pkl_path (_type_): _description_
            max_workers (_type_): _description_
        """
        length = len(dataset)
        print(f'ALL {length} samples to save')
        all_sample_list = [i for i in dataset]
        segment_list = oso.split_list(all_sample_list,max_workers)
        temp_file_path_list= [f'temp{i}.pkl.gz' for i in range(max_workers)]
        
        params_list = list(zip(segment_list,temp_file_path_list))
        proc_bar = tqdm(length,desc="Saving to pkl:")
        
        def save_sample(part_sample_list, file_path):
            for x,y in part_sample_list:
                proc_bar.update(1)
                with gzip.open(file_path, 'ab') as f:
                    
                    pickle.dump((np.asanyarray(x),np.asanyarray(y)), f)
        
        
               
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:  
            futures = [executor.submit(save_sample, part_sample_list, pkl_path) for part_sample_list,pkl_path in params_list]

            concurrent.futures.wait(futures)

        oso.merge_files(temp_file_path_list,pkl_path)
        for i in temp_file_path_list:
            oso.delete_file(i)
        
        proc_bar.close()
        print(f'dataset saved to {pkl_path}')
        print('Notice: when you need to open it, please include your dataset defination in open code')

    @classmethod
    def save_dataset_to_npz(cls,
                            dataset,
                            npz_path:str
                            ):
        print(f'ALL {len(dataset)} samples to save')
        
        proc_bar = tqdm(len(dataset),"changing dataset type to ndarray:")
        X_list = []
        y_list = []
        for X,y in dataset:
            X_list.append(X)
            y_list.append(y)
            proc_bar.update(1)
        X = np.asanyarray(X_list)
        y = np.asanyarray(y_list)
        np.savez_compressed(npz_path,X = X,y = y)
        proc_bar.close()
        print(f'dataset saved to {npz_path}')
        print('Notice: when you need to open it, please include your dataset defination in open code')
    
    @classmethod
    def save_dataset_to_hdf5(cls,
                             dataset:Dataset,
                             hdf5_path:str):
        print(f'ALL {len(dataset)} samples to save')
        proc_bar = tqdm(total=len(dataset), desc="Saving to HDF5:")
        for X,y in dataset:
            X = np.asanyarray(X)
            y = np.asanyarray(y) 
            
            X_shape = X.shape
            y_shape = y.shape
            X_dtype = X.dtype
            y_dtype = y.dtype
            break

        with h5py.File(hdf5_path,'w') as hf:
            X_group = hf.create_group('X')
            y_group = hf.create_group('y')
            X_group.attrs['shape'] = f'{X_shape}'
            X_group.attrs['dtype'] = f'{X_dtype}'
            y_group.attrs['shape'] = f'{y_shape}'
            y_group.attrs['dtype'] = f'{y_dtype}'
            hf.attrs['length'] = len(dataset)
            
            if X_shape ==() and y_shape == ():
                for i,sample in enumerate(dataset):
                    X,y = sample
                    X = np.asanyarray(X).reshape(-1)
                    y = np.asanyarray(y).reshape(-1)
                    X_group.create_dataset(f'{i}',data=X)
                    y_group.create_dataset(f'{i}',data=y)
                    proc_bar.update(1)
            elif X_shape == () and y_shape != ():
                for i,sample in enumerate(dataset):
                    X,y = sample
                    X = np.asanyarray(X).reshape(-1)
                    y = np.asanyarray(y)
                    X_group.create_dataset(f'{i}',data=X)
                    y_group.create_dataset(f'{i}',data=y)
                    proc_bar.update(1)
            elif X_shape != () and  y_shape == ():
                for i,sample in enumerate(dataset):
                    X,y = sample
                    X = np.asanyarray(X)
                    y = np.asanyarray(y).reshape(-1)
                    X_group.create_dataset(f'{i}',data=X)
                    y_group.create_dataset(f'{i}',data=y)
                    proc_bar.update(1)
            else:
                for i,sample in enumerate(dataset):
                    X,y = sample
                    X = np.asanyarray(X)
                    y = np.asanyarray(y)
                    X_group.create_dataset(f'{i}',data=X)
                    y_group.create_dataset(f'{i}',data=y)
                    proc_bar.update(1)

        proc_bar.close()
        print(f"Dataset saved to HDF5 file {hdf5_path}.")
        
    @classmethod
    def get_dataset_from_hdf5(cls,
                             hdf5_path:str,
                             open_mode:str = 'r')->Dataset:
        
        X_list = []
        y_list = []
        with h5py.File(hdf5_path,open_mode) as hf:
            length = hf.attrs['length']
            
            print(f'All {length} samples to read')
            proc_bar = tqdm(length,'Load hdf5 file:')

            for i in range(length):
                
                X_list.append(hf['X'][f'{i}'][:])
                y_list.append(hf['y'][f'{i}'][:])
                proc_bar.update(1)
            proc_bar.close()
            print('Reading over')
            
        X = np.asanyarray(X_list)
        y = np.asanyarray(y_list)
        return TensorDataset(torch.from_numpy(X),torch.from_numpy(y))
       
    @classmethod
    def get_dataset_from_npz(cls,
                             npz_dataset_obj:Dataset,
                             npz_path:str):
        oso.get_current_datetime(True)
        data = np.load(npz_path)
        X = torch.from_numpy(data['X'])
        y = torch.from_numpy(data['y'])
        dataset = TensorDataset(X,y)
        print(f'{npz_path}   npz dataset length is {len(dataset)}')
        oso.get_current_datetime(True)
        return dataset

    
    @classmethod
    def get_dataset_from_pkl(cls,
                             pkl_dataset_obj:Dataset,
                             pkl_save_path:str,
                             open_mode:str = 'rb')->Dataset:
        def load_data(file_path):
            with gzip.open(file_path, open_mode) as f:
                data = []
                length = pickle.load(f)
                proc_bar = tqdm(length,desc=f"Reading from {file_path}:")
                while True:
                    try:
                        sample = pickle.load(f)
                        data.append(sample)
                        proc_bar.update(1)
                    except EOFError:
                        proc_bar.close()
                        print('Finish reading')
                        break
            return data
        loaded_data = load_data(pkl_save_path)
        x = np.asanyarray([sample[0] for sample in loaded_data])
        y = np.asanyarray([sample[1] for sample in loaded_data])
        X = torch.tensor(x)
        y = torch.tensor(y)

        d = TensorDataset(X,y)
        return d
    
    
    @classmethod
    def get_path_info_from_yaml(cls,yaml_path:str)->tuple:
        """get path info from yaml 

        Args:
            yaml_path (str): _description_

        Returns:
            train_path,val_path,weights_save_path
        """
        info =Data.get_file_info_from_yaml(yaml_path)
        train_path = info['train_path']
        val_path = info['val_path']
        weights_save_path = info['weights_save_path']
        return train_path,val_path,weights_save_path
        
         
class dataset_hdf5(Dataset):
    def __init__(self,hdf5_path:str) -> None:
        super().__init__()
        
        self.path = hdf5_path
        with h5py.File(hdf5_path,'r') as hf:
            self.length = hf.attrs['length']
            
    def __len__(self):
        return self.length
    
    def __getitem__(self, index)->tuple:
        with h5py.File(self.path,'r') as hf:
            X = hf['X'][f'{index}'][:]
            y = hf['y'][f'{index}'][:]
        return torch.from_numpy(X),torch.from_numpy(y)
            
class dataset_pkl(Dataset):
    def __init__(self,pkl_path:str) -> None:
        super().__init__()
        
        self.path = pkl_path
        with gzip.open(pkl_path,'rb') as f:
            self.length = pickle.load(f)
            self.position = f.tell()
        
            
    def __len__(self):
        return self.length
    
    def __getitem__(self, index)->tuple:
        with gzip.open(self.path,'rb') as f:
            f.seek(self.position)
            X,y = pickle.load(f)
            self.position = f.tell()
        
        return torch.from_numpy(X),torch.from_numpy(y)
   
            
        

class Mouse_trajectory_capture:
    '''
    generate a series of points by the mouse trajectory\n
    curve_nums decides how many curves can be drawn in one ax\n
    use start_draw to get return in list
    
    
    '''
    def __init__(self,curve_nums:int=2, xlim:tuple=(0,1),ylim:tuple=(0,1),figsize:tuple=(15,15)):
        self.f_obj=Data.plt_figure(figsize=figsize)
        
        self.ax = self.f_obj.figure.add_subplot()
        self.ax.set_xlim(xlim[0],xlim[1])
        self.ax.set_ylim(ylim[0],ylim[1])
        self.points = []
        self.lines=[]
        #red, green, blue, cyan, magenta, yellow, black, white
        self.colors=['r','g','b','c','m','y','k','w']
        self.labels=[str(i) for i in range(curve_nums)]
        init_count=0
        for i in range(curve_nums):
            self.points.append([])
            if init_count==len(self.colors):
                init_count=0
            
            self.lines.append(self.ax.plot([], [], marker='o', color=self.colors[init_count], markersize=2, label=self.labels[init_count])[0])
            init_count+=1
        #connection id for specified event
        self.cid_press = None
        self.cid_motion = None
        self.cid_release = None
        self.count=0
        self.curve_nums=curve_nums
        self.ax.set_title("Press and drag to draw. Right-click to finish.")
        self.ax.legend()

        self.connect()
    
    def start_draw(self)->list:
        '''
        return list of curve_points
        '''
        plt.show()
        for index,i in enumerate(self.points):
            self.points[index]=np.array(self.points[index])
        return self.points
    
    def connect(self):
        self.cid_press = self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_motion = self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_release = self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)

    def on_press(self, event):
        # Left mouse button
        if event.button == 1:  
            self.points[self.count].append((event.xdata, event.ydata))
            self.update_plot()
        # Right mouse button
        elif event.button == 3:  
            self.disconnect()

    def on_motion(self, event):
        # Left mouse button and inside the axes
        if event.button == 1 and event.inaxes:  
            self.points[self.count].append((event.xdata, event.ydata))
            self.update_plot()

    def on_release(self, event):
        # Left mouse button
        if event.button == 1:
            self.count+=1
            if self.count== self.curve_nums:
                 
                self.disconnect()

    def update_plot(self):
        if self.points[self.count]:
            x, y = zip(*self.points[self.count])
            self.lines[self.count].set_data(x, y)
            self.ax.figure.canvas.draw()
            
    def disconnect(self):
        if self.cid_press is not None:
            self.ax.figure.canvas.mpl_disconnect(self.cid_press)
            self.ax.figure.canvas.mpl_disconnect(self.cid_motion)
            self.ax.figure.canvas.mpl_disconnect(self.cid_release)
            self.cid_press = None
            self.cid_motion = None
            self.cid_release = None
            



def mouse_click_capture(xlim:tuple|None=(0,1),ylim:tuple|None=(0,1)):
    f=Data.plt_figure()
    ax=f.figure.add_subplot()
    if isinstance(xlim,tuple):
        ax.set_xlim(xlim[0],xlim[1])
    if isinstance(ylim,tuple):
        
        ax.set_ylim(ylim[0],ylim[1])
    # Capture mouse trajectory
    points = plt.ginput(n=-1, timeout=0, show_clicks=True)
    
    # Close the plot
    plt.close()

    return points    



if __name__=="__main__":
    path="../pth_folder/test.pth"
    def func0(x):
        return np.power(x,2)

    def func1(x):
        return np.power(x,3)

    
    model1=simple_2classification(feature_nums=4,feature_expand=True,correlation=[[[0],[0],func0],[[1],[1],func0],[[0],[2]],[[1],[3]]])
    checkpoints=torch.load(path)
    model1.load_state_dict(checkpoints)
    print(model1.state_dict())
    
    hp_pts=Data.get_hyperplane_pts(model1,pt_nums=100)
    f=Data.plt_figure()
    ax_index=f.plt_point(x=hp_pts[:,0],y=hp_pts[:,1],xlim=(-5,5),ylim=(-5,5))
    plt.show()
    
    
    

    

    
    
    
    
    
    
    
    
    