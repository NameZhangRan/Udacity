#-*-coding:utf-8-*-

# TODO 创建一个 4*4 单位矩阵

# 实现：
# 1.定义一个 n*n 的单位矩阵函数
# 2.实现一个 4*4 的单位矩阵

I = [0]*4
#print I
#预览[0, 0, 0, 0]
I[0] = [0]*4
#print I
#预览[[0, 0, 0, 0], 0, 0, 0]
I[0][0] = 1
#print I
#预览[[1, 0, 0, 0], 0, 0, 0]
I = [0]*4
#print I
#预览[0, 0, 0, 0]
for i in range(0,4):
    I[i] = [0]*4
    I[i][i] = 1
print I

#样本：
I = [0]*4
for i in range(0,4):
    I[i] = [0]*4
    I[i][i] = 1
print I

#泛化：
def identity_matrix(n):
    I = [0] * n
    for i in range(0, n):
        I[i] = [0]*n
        I[i][i] = 1
    return I
print identity_matrix(4)