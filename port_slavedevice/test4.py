number = 527280139

# 将整数转换为4字节的大端存储的字节序列
byte_sequence_big = number.to_bytes(4, byteorder='big')

print(byte_sequence_big)
number_big = int.from_bytes(byte_sequence_big, byteorder='little')

print(number_big)

# 将整数转换为4字节的小端存储的字节序列
byte_sequence_little = number.to_bytes(4, byteorder='little')

print(byte_sequence_little)
number_little = int.from_bytes(byte_sequence_little, byteorder='big')

print(number_little)
