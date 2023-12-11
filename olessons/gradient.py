import numpy as np

import sys
sys.path.append('../utils_network/')
from utils_network.data import *


if __name__=="__main__":
    
    
    
    import numpy as np

    def power2d(x):
        return np.power(x[0], 2) + np.power(x[1], 2)

    def gradient_power2d(x):

        gradient = np.array([2 * x[0], 2 * x[1]])
        return gradient

    
    def normal_pdf1d(x:np.ndarray):
        return (1/math.sqrt(2*math.pi))*np.exp(-0.5*np.power(x,2))
    def normal_pdf2d(x:np.ndarray):
        
        return (1/math.sqrt(2*math.pi))*np.exp(-0.5*(np.power(x[0],2)+np.power(x[1],2)))
    def gradient_descent(initial_x, learning_rate, num_iterations,point_data:Data.random_point):
        
        current_x = initial_x
        point_data.scatter3d()
        for i in range(num_iterations):
            gradient = gradient_power2d(current_x)

            current_x = current_x - learning_rate * gradient
            
            
            new_point=np.array([current_x[0],current_x[1],power2d(current_x)]).reshape(3,1)
            point_data.scatter3d(new_pt=new_point)
            print(f"Iteration {i+1}: x = {current_x}, Power2D(x) = {power2d(current_x)}")
            
        return current_x
    
    data_maker=Data(10)
    point_data=data_maker.random_point(3,5000,(-3,3),[[[0,1],[2],power2d]])
    
    init_pt=np.array([2,2]).reshape(2,1)
    learning_rate=0.1
    iters=10
    
    gradient_descent(init_pt,learning_rate,iters,point_data)
    