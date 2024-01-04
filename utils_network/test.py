import torch

# 创建一个 4x4 的输入矩阵
input_matrix = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                             [5.0, 6.0, 7.0, 8.0],
                             [9.0, 10.0, 11.0, 12.0],
                             [13.0, 14.0, 15.0, 16.0]])
input_matrix = torch.tensor([[1,2],
                             [3,4]],dtype=torch.float32)

# 将输入矩阵调整为 1x1 的输出
adaptive_avg_pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
output = adaptive_avg_pooling(input_matrix.view(1, 1, 2, 2))  # 需要将输入调整为 (batch_size, channels, height, width)

# 输出结果
print(output)
