import cv2
import numpy as np
import time

# 初始化变量
y_values = []
y = 0.0
x_values = 0

# 创建窗口
cv2.namedWindow('Dynamic Plot', cv2.WINDOW_NORMAL)

while True:
    # 更新数据的函数
    y += 0.1
    y_values.append(y)

    # 创建空白图像
    img = np.ones((300, 800, 3), dtype=np.uint8) * 255

    # 绘制曲线
    for i in range(len(y_values) - 1):
        cv2.line(img, (i * 8, int(200 - y_values[i] * 20)),
                 ((i + 1) * 8, int(200 - y_values[i + 1] * 20)), (0, 0, 255), 2)

    # 显示图像
    cv2.imshow('Dynamic Plot', img)

    # 每隔0.2秒更新一次
    time.sleep(0.2)

    # 检测键盘输入，按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    x_values += 1

# 销毁窗口
cv2.destroyAllWindows()
