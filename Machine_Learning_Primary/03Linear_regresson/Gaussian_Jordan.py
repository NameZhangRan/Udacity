#-*-coding:utf-8-*-
# Gaussian Jordan 消元法求解 Ax = b

"""
A = [[1, 3, 1],
     [2, 1, 1],
     [2, 2, 1]]
b = [11, 8, 10]
样本解：
x = [ [1],
      [2],
      [4] ]

"""

# 需要用到的其它函数(本仓库的其它文件都有示例)
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


# 开始定义：
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

        print 'A:'
        print A
        Ab_T = transpose(Ab)

        # TODO 调试
        print '转置A:'
        print Ab_T

        # 定义当前列系数绝对值的独立列表cd/c绝对值合并列c_abs/最大绝对值合并列c_max
        cd = []
        # 最大值索引
        c_max_index = []
        c_max = []
        for index in range(len(Ab_T[0])):
            cd.append(Ab_T[index])
            print 'cd:'
            print cd
            for index_2 in range(index, len(Ab_T[0])):
                c_abs = [abs(value) for value in Ab_T[index][index:]]
            # TODO 调试
            print 'c_abs:'
            print c_abs
            c_max.append(max(c_abs))

            # TODO 调试
            print 'c_max:'
            print c_max

            # 获取系数
            c_max_index.append(c_abs.index(c_max[index]) + index)
            print 'c_max_index:'
            print c_max_index

        if min(c_max) <= epsilon:
            return None
        else:
            # TODO 调试
            print '测试判断奇异矩阵成功'

            # 开始实现Gaussian Jordan核心部分：
            # 返回列向量 x 使得 Ax = b
            # 把绝对值最大行交换到对角线行
            print '换行前:'
            print Ab
            for index in range(col_A):
                swapRows(Ab, index, c_max_index[index])
            print '换行后:'
            print Ab

            # 把列对角线元素缩放为1
            for index in range(col_A):
                scaleRow(Ab, index, 1. / Ab[index][index])
                print '对角线系数为1：'
                print Ab
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
# test

A = [[1, 3, 1],
     [2, 1, 1],
     [2, 2, 1]]
b = [[11],
     [8],
     [10]]

print gj_Solve(A, b)


