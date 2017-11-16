#-*-coding:utf-8-*-

# 初等行变换

# 1交换两行
# 2把某行乘以一个非零常数
# 3把某行加上另一行的若干倍

# TODO r1 <---> r2
# 1直接修改参数矩阵，无返回值
# def swapRows(M, r1, r2):
#    pass
# 样本
# 示例

A = [[1,1,1],
     [2,2,2],
     [3,3,3],
     [4,4,4]]
#print A

x = A[0]
y = A[1]

A[0] = y
A[1] = x
#print A

# 泛化

def swapRows(M, r1, r2):
    M[r1], M[r2] = M[r2], M[r1]

    return M
#print swapRows(A, 0, 1)

# TODO r1 <--- r1 * scale， scale!=0
# 2直接修改参数矩阵，无返回值
# def scaleRow(M, r, scale):
#     pass

def scaleRow(M, r, scale):
    if scale == 0:
        raise ValueError
    else:
        M[r] = [i * scale for i in M[r]]

    return M

M = [[1, 1, 1],
     [2, 2, 2],
     [3, 3, 3],
     [4, 4, 4]]

#print scaleRow(M, 3, 2)


# TODO r1 <--- r1 + r2*scale
# 3直接修改参数矩阵，无返回值
# def addScaledRow(M, r1, r2, scale):
#     pass

def addScaledRow(M, r1, r2, scale):
    x = [i * scale for i in M[r2]]
    M[r1] = [a + b for a, b in zip(M[r1], x)]

    return M






M = [[1, 1, 1],
     [2, 2, 2],
     [3, 3, 3],
     [4, 4, 4]]
print addScaledRow(M, 0, 2, 3)




