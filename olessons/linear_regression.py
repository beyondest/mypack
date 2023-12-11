import numpy as np
import sympy
import scipy
import random 
import time
import matplotlib.pyplot as plt
import math
import scipy
import sys
sys.path.append('../utils_network/')
from utils_network.data import Data

    
         

           
           
if __name__=='__main__':
    def func0(x:np.ndarray)->np.ndarray:
        return np.power(x,2)
    
    def func1(x:np.ndarray)->np.ndarray:
        return np.exp(x)
    def func2(x:np.ndarray):
        return np.log(np.abs(x))
    def func3(x):
        return np.sin(x)
    def func4(x):
        return np.power(x,3)
    data_maker=Data(seed=int(time.perf_counter()))
    point_data=data_maker.random_point(2,1000,(-4,4),
                            [[[0],[0]],[[0],[1],func1]],
                            [[[1],"normal",(0,5)]]
                            )
    
    
    #Ax=y
    A=data_maker.make_taylor_basis(point_data.out[0],order=3)
    y=np.copy(point_data.out[1]).reshape(-1,1)

    t1=time.perf_counter()
    #A@pinv(A.T@A)@A.Ty=x_hex
    
    x_hex=np.linalg.pinv(A.T@A)@A.T@y
    y_hex=A@x_hex
    t2=time.perf_counter()
    
    print("timeis",t2-t1)
    
    #predict using best coefficient_array:x_hex;oder must be same
    axis_predict=np.linspace(4,6,1000)
    B=data_maker.make_taylor_basis(axis_predict,order=3)
    y_predict=B@x_hex
    
    
    
    figure=plt.figure(figsize=(10,10))
    figure.canvas.mpl_connect("key_press_event",point_data.key_close)
    ax=figure.add_subplot(111)
    
    ax.plot(point_data.out[0],y_hex,color='red',linewidth=1)
    ax.scatter(point_data.out[0],point_data.out[1],color='blue')
    ax.plot(axis_predict,y_predict,color='green',linewidth=1)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()