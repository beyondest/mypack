import numpy as np
from sympy import symbols,diff,integrate,lambdify,hessian,sqrt,Pow
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import inspect
from utils_network.data import *


import math


class field:
    def __init__(self,func,dim,scope:list,num:int=100) -> None:
        x_list=[]
        for i in range(dim):
            x_list.append(np.linspace(scope[0],scope[1],num))
        grid_list=np.meshgrid(*x_list)



        self.scalar_field=func(*grid_list)
    
    def get_vector_field(self):
        '''return norm2_vectorfield'''
        grad_list=np.gradient(self.scalar_field)
        f=lambda x,y:(x**2+y**2)**0.5
        self.norm2vector_field=f(*grad_list)
        return self.norm2vector_field
    
    
class myfunc:
    
     
    def cov_show(self):
        x=np.arange(0,10,0.1)
        y=np.random.randn(len(x))
        y=x**2+y
        cov=np.cov(x,y)
        print(cov)
        plt.imshow(cov,cmap="gray",interpolation="nearest")
        plt.show()
        
    def hessian_show(self):

        x,y = symbols('x y')
        f = x**3 + y**4

        H = hessian(f, [x, y])
        print("Hessian Matrix:")
        print(H)
    
    @classmethod
    def relu(cls,x:np.ndarray)->np.ndarray:
        return np.maximum(0,x)
    
    @classmethod
    def sigmoid(cls,x:np.ndarray)->np.ndarray:
        
        return 1/(1+np.exp(-x))
    
    @classmethod
    def generate_np1d(cls,isdiff:bool=False,show:bool=True):
        '''
        change source code here, to show source code generated by sympy
        '''
        x=symbols('x')
        
        
        #####################for change###################
        f=sqrt(x**2-x**4)
        
        
        ####################################################
        
        
        
        if isdiff:
            out=diff(f,x)
        else:
            out=f
        out=lambdify(x,out,'numpy')
        src_diff_f_np=inspect.getsource(out)
        if show:
            print(src_diff_f_np)
        return out
    def normal_pdf1d(x:np.ndarray):
        return (1/math.sqrt(2*math.pi))*np.exp(-0.5*np.power(x,2))      

    def normal_pdf2d(x:np.ndarray)->np.ndarray:
        '''
        input x.shape=(samples,dims),dim0 is x,dim1 is y, ruturn z 
        '''
        return (1/math.sqrt(2*math.pi))*np.exp(-0.5*(np.power(x[:,0],2)+np.power(x[:,1],2)))


   

def map_value(value, ori_scope:tuple, target_scope:tuple):
    
    if value > ori_scope[1] or value <ori_scope[0]:
        raise TypeError("Input value out of scope")
    from_range = ori_scope[1] - ori_scope[0]
    to_range = target_scope[1] - target_scope[0]
    
    scaled_value = (value - ori_scope[0]) / from_range
    result = target_scope[0] + scaled_value * to_range
    
    return result


if __name__=='__main__':
    pass