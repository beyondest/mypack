import msvcrt

print("Press any key to exit...")

while not msvcrt.kbhit():
    # 这里可以放入程序的其他逻辑
    pass
    # 例如，可以在这里处理其他任务

# 从键盘获取按下的键值
key = msvcrt.getch()

print(f"Key pressed: {key}")
print("Exiting the program.")
