import gzip
import struct
from PIL import Image
import numpy as np

def read_idx3_ubyte(file_path):
    with gzip.open(file_path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        image_data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
    return image_data

def convert_to_jpg(images, output_folder):
    for i, image_data in enumerate(images):
        image = Image.fromarray(image_data, 'L')
        image.save(f"{output_folder}/mnist_{i}.jpg")

# 文件路径
mnist_file_path = './data/MNIST/raw/t10k-images-idx3-ubyte.gz'
output_folder = './out2'


# 读取 MNIST 图像数据
mnist_images = read_idx3_ubyte(mnist_file_path)

# 转换为 JPG 图像
convert_to_jpg(mnist_images, output_folder)
