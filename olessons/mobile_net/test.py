import gzip
import pickle
from torchvision import datasets
import concurrent.futures

# 定义数据集
dataset = datasets.FakeData(transform=None)
pkl_save_path = './dataset.pkl.gz'
open_mode = 'wb'  # 设置为二进制模式

print(f'ALL {len(dataset)} samples to save')

def save_dataset(samples):
    X, y = zip(*samples)
    X = [x.numpy() for x in X]
    y = [yi.numpy() for yi in y]
    
    with gzip.open(pkl_save_path, open_mode) as file:
        pickle.dump({'X': X, 'y': y}, file)

# 收集所有样本
all_samples = list(dataset)

# 将所有样本一次性保存到同一个文件
with concurrent.futures.ProcessPoolExecutor() as executor:
    executor.map(save_dataset, [all_samples])

print(f'dataset saved to {pkl_save_path}')
print('Notice: when you need to open it, please include your dataset definition in open code')
