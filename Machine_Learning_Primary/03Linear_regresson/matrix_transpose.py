#-*-coding:utf-8-*-

# TODO 计算矩阵的转置
# 使用zip函数的变型unzip转换矩阵zip(*M)
# 把转换成的tuple再格式为list

# def transpose(M):
#     return None

M = [[1.123456, 2.123456, 3.123456],
     [4.123456, 5.123456, 6.123456]]

def transpose(M):
    return [list(row) for row in zip(*M)]

print transpose(M)