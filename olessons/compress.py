import cv2
import numpy as np
import sys
sys.path.append('..')
import img.img_operation as imo
path='./res/armorred.png'
img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)


threshold=50
def compress_img(img:np.ndarray, k:int=50):
    '''
    return svd 
    '''
    

    compressed_channels = []
    for channel in cv2.split(img):
        U, S, Vt = np.linalg.svd(channel, full_matrices=False)
        U_k = U[:, :k]
        S_k = np.diag(S[:k])
        Vt_k = Vt[:k, :]
        compressed_channel = U_k@S_k@Vt_k
        compressed_channel = np.round(compressed_channel).astype(np.uint8)
        compressed_channels.append(compressed_channel)

    compressed_image = cv2.merge(compressed_channels)
    return compressed_image

    
out=compress_img(img,k=threshold)

imo.cvshow(out)

