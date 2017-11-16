#-*-coding:utf-8-*-

# 求解方程  X_T*X*h = X_T*Y , 计算线性回归的最佳参数 h
# X Y h 数据在loss_function_gradient.ipynb

# TODO 实现线性回归
# 参数：(x,y) 二元组列表
# 返回：m，b

# 定义矩阵的行列
def shape(M):
    i = len(M)
    j = len(M[0])
    return i,j
# 构造增广矩阵，假设Aa，bb行数相同
def augmentMatrix(A, b):
    for i in range(len(b)):
        A[i].append(b[i][0])
    return A
# 矩阵的转置
def transpose(M):
    return [list(row) for row in zip(*M)]
def swapRows(M, r1, r2):
    M[r1], M[r2] = M[r2], M[r1]
    pass
def scaleRow(M, r, scale):
    if scale != 0:
        M[r] = [M_value * scale for M_value in M[r]]
    else:
        raise ValueError
def addScaledRow(M, r1, r2, scale):
    if scale != 0:
        row2 = [M_value * scale for M_value in M[r2]]
        M[r1] = [x + y for x, y in zip(M[r1], row2)]
        return M
def matxMultiply(A, B):
    A_row, A_col = shape(A)
    B_row, B_col = shape(B)
    if not A_col == B_row:
        return None
    else:

        C = []
        for A_row_num in range(A_row):
            C_row = []
            for B_col_num in range(B_col):
                C_row.append(sum(A[A_row_num][e] * B[e][B_col_num] for e in range(A_col)))
            C.append(C_row)
    return C
# Gaussian Jordan
def gj_Solve(A, b, decPts=4, epsilon = 1.0e-16):
    # 解为GJ
    GJ = []
    # 检查A，b是否行数相同
    row_A, col_A = shape(A)
    if row_A != len(b):
        return None
    # 构造增广矩阵
    else:
        Ab = augmentMatrix(A, b)
        # 判断奇异矩阵：
        # 首先，寻找当前列c中
        # 对角线以及对角线以下所有元素（行 c~N）的
        # 绝对值的最大值是否大于零
        # 所以，对Ab进行转置，方便遍历元素
        Ab_T = transpose(Ab)
        # 定义当前列系数绝对值的独立列表cd/c绝对值合并列c_abs/最大绝对值合并列c_max
        cd = []
        # 最大值索引
        c_max_index = []
        c_max = []
        for index in range(len(Ab_T[0])):
            cd.append(Ab_T[index])
            for index_2 in range(index, len(Ab_T[0])):
                c_abs = [abs(value) for value in Ab_T[index][index:]]
            c_max.append(max(c_abs))
            # 获取系数
            c_max_index.append(c_abs.index(c_max[index]) + index)
        if min(c_max) <= epsilon:
            return None
        else:
            # 开始实现Gaussian Jordan核心部分：
            # 返回列向量 x 使得 Ax = b
            # 把绝对值最大行交换到对角线行
            for index in range(col_A):
                swapRows(Ab, index, c_max_index[index])
            # 把列对角线元素缩放为1
            for index in range(col_A):
                scaleRow(Ab, index, 1. / Ab[index][index])
                # 多次addScaledRow行变换，将列的其他元素消为0
                for e in range(col_A):
                    if e != index:
                        addScaledRow(Ab, e, index,
                                     -1 * Ab[e][index] / Ab[index][index])
            y = shape([transpose(Ab)[-1]])
            for answer in [transpose(Ab)[-1]]:
                for i in range(y[1]):
                    GJ.append([round(answer[i], decPts)])
            return GJ
            print GJ

# 定义线性回归方程
print '定义线性回归方程'

def linearRegression(points):
    # 构建 Ax = b 的线性方程
    X = [[points[i][0], 1] for i in range(len(points))]
    #print X
    Y = [[points[i][1]] for i in range(len(points))]
    #print Y
    X_T = transpose(X)
    #print X_T
    A = matxMultiply(X_T, X)
    #print A
    b = matxMultiply(X_T, Y)
    #print b

    m, b = (i[0] for i in gj_Solve(A, b, decPts=4, epsilon=1.0e-16))
    return m, b

    #m, b = gj_Solve(A, b)
    print m, b

#points = [[1,2], [3,4], [4,7]]
#linearRegression(points)

# 测试你的线性回归实现
print '测试你的线性回归实现'

# 构造线性函数

m = 2
b = 3

# 构造100个线性函数上的点，加上适当的高斯噪音

import random

x_origin = [random.uniform(1,10) for i in range(100)]
Y_origin = [i * m + b for i in x_origin]

x_gauss = [random.gauss(0, 0.1) for i in range(100)]
y_gauss = [random.gauss(0, 0.1) for i in range(100)]

xx = [x + y for x,y in zip(x_origin, x_gauss)]
yy = [x + y for x,y in zip(Y_origin, y_gauss)]

points = [(x,y) for x,y in zip(xx, yy)]

# 对这100个点进行线性回归，将线性回归得到的函数和原线性函数比较

m_gauss, b_gauss = linearRegression(points)

print '原始m, b '
print m, b

print '处理后的m, b'
print m_gauss, b_gauss