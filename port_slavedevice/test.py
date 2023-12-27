import cv2
import numpy as np

a = [12,12]

# 回调函数
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 获取传递给回调函数的额外参数
        param[0]+=1
        
        # 在点击位置绘制一个圆
        cv2.circle(img, (x, y), 10, (0, 255, 0), -1)
        
        # 在图像上显示文本
        
        # 更新图像显示
        cv2.imshow('Image with Circle', img)

# 创建一个黑色图像
img = np.zeros((512, 512, 3), np.uint8)

# 显示原始图像
cv2.imshow('Image with Circle', img)

# 设置鼠标点击事件的回调函数，并传递额外参数
param =a
cv2.setMouseCallback('Image with Circle', click_event, param)

# 等待用户按下 ESC 键退出
while True:
    print(param[0])
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # 按下 ESC 键退出
        break

# 关闭窗口
cv2.destroyAllWindows()
