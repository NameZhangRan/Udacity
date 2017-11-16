#-*-coding:utf-8-*-

# TODO 构造增广矩阵，假设A，b行数相同
# 遍历A矩阵每行并appen对应b的每行
A = [ [1,2],
      [3,4],
      [5,6] ]
b = [ 1,
      1,
      1 ]

#样本
A[0].append(b[0])
A[1].append(b[1])
print A

#泛化

row = len(b)
for i in range(row):
    A[i].append(b[i])
print A

def augmentMatrix(A, b):
    for i in range(len(b)):
        A[i].append(b[i])
    return A

print A
