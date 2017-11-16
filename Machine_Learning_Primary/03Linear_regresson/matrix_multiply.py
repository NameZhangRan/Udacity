#-*-coding:utf-8-*-

# TODO 计算矩阵乘法 AB，如果无法相乘则返回None

# 设C=AB，首先判断AB矩阵是否可以相乘
# 根据矩阵乘法的公式算出C

def shape(M):
    i = len(M)
    j = len(M[0])
    return i,j

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
                C_row.append(sum(A[A_row_num][e]*B[e][B_col_num] for e in range(A_col)))
            C.append(C_row)
    return C



A = [[1, 2, 3],
     [4, 5, 6]]
B = [[1, 4],
     [2, 5],
     [3, 6]]

print matxMultiply(A, B)
