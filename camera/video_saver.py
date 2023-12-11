#coding=utf-8
import numpy as np
import math
import cv2
import control as con

class Videosaver:
    def __init__(self) -> None:
        pass
    
    @classmethod
    def byphone(cls,path:str="rtsp://127.0.0.1:8080/h264_pcm.sdp"):
        '''
        no img_processing
        '''
        vd=cv2.VideoCapture(path)
        while True:
            ret, frame = vd.read()  
            if not ret:
                break
            cv2.imshow('press esc to break', frame)  
            if cv2.waitKey(1) & 0xFF ==27:
                break
        vd.release()  
        cv2.destroyAllWindows() 
        
    def bymindvision(cls):
        pass
    
    
    
    
    
    
if __name__=="__main__":
    Videosaver.byphone()
