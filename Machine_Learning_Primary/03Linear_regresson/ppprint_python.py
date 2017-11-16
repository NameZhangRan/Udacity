#-*-coding:utf-8-*-
def shape(M):
    i = len(M)
    j = len(M[0])
    return i,j

B = [[1,2,3,5],
     [2,3,3,5],
     [1,2,5,1]]
import pprint
pp = pprint.PrettyPrinter(indent=1, width=20)

print '测试1.2 返回矩阵的行和列'
pp.pprint(B)
print(shape(B))