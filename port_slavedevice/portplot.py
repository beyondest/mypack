
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from itertools import count
from matplotlib.animation import FuncAnimation
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# 初始化变量
y_values = []
y = 0.0
x_values = count()

# 创建图形和坐标轴
fig, ax = plt.subplots()

# 设置坐标轴标签
ax.set_xlabel('Time')
ax.set_ylabel('y Value')

# 设置连线图形式
line, = ax.plot([], [], label='y values', marker='o', linestyle='-')

# 更新数据的函数
def update(frame):
    global y
    y += 0.1
    y_values.append(y)
    
    # 更新图形数据
    line.set_data(list(x_values), y_values)

    # 设置坐标轴范围
    ax.relim()
    ax.autoscale_view()
    
    # 手动刷新图形
    fig.canvas.draw_idle()

# 使用FuncAnimation周期性更新数据
ani = FuncAnimation(fig, update, interval=200, save_count=100)

# 创建Qt应用程序和主窗口
app = QApplication([])
window = QMainWindow()
window.setGeometry(100, 100, 800, 600)

# 创建一个布局并将图形添加到其中
layout = QVBoxLayout()
canvas = FigureCanvas(fig)
layout.addWidget(canvas)

# 创建一个QWidget，将布局设置为主窗口的中央部件
widget = QWidget()
widget.setLayout(layout)
window.setCentralWidget(widget)

# 显示图形
window.show()

# 启动Qt应用程序的事件循环
app.exec_()
