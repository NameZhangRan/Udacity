#-*-coding:utf-8-*-
# 返回矩阵的行列

def shape(M):
    i = len(M)
    j = len(M[0])
    return i,j

M = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9] ]

print shape(M)