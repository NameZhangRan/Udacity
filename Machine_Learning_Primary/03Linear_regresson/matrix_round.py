#-*-coding:utf-8-*-

# 使矩阵里的每个元素四舍五入到特定位数
# 遍历每个元素使用round进行四舍五入

M = [[1.123456, 2.123456],
     [3.123456, 4.123456],
     [5.123456, 6.123456]]


for x in M:
    for y in range(len(x)):
        x[y] = round(x[y], 4)

#print M

# 泛化

def matxRound(M, decPts = 4):
    for x in M:
        for y in range(len(x)):
            x[y] = round(x[y], decPts)
    pass

print M