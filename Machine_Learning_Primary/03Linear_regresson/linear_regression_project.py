
# coding: utf-8

# # 1 矩阵运算
# 
# ## 1.1 创建一个 4*4 的单位矩阵

# In[1]:


# 这个项目设计来帮你熟悉 python list 和线性代数
# 你不能调用任何python库，包括NumPy，来完成作业

A = [[1,2,3], 
     [2,3,3], 
     [1,2,5]]

B = [[1,2,3,5], 
     [2,3,3,5], 
     [1,2,5,1]]

#TODO 创建一个 4*4 单位矩阵

# 1.定义一个 n*n 的单位矩阵函数
# 2.实现一个 4*4 的单位矩阵

def identity_matrix(n):
    I = [0] * n
    for i in range(0, n):
        I[i] = [0]*n
        I[i][i] = 1
    return I
print identity_matrix(4)


# ## 1.2 返回矩阵的行数和列数

# In[2]:


# TODO 返回矩阵的行数和列数

# 使用len直接获取矩阵的行和列数

def shape(M):
    i = len(M)
    j = len(M[0])
    return i,j


# ## 1.3 每个元素四舍五入到特定小数数位

# In[3]:


# TODO 每个元素四舍五入到特定小数数位
# 直接修改参数矩阵，无返回值

# 遍历每个元素使用round进行四舍五入

def matxRound(M, decPts = 4):
    for x in M:
        for y in range(len(x)):
            x[y] = round(x[y], decPts)
    pass


# ## 1.4 计算矩阵的转置

# In[4]:


# TODO 计算矩阵的转置

# 使用zip函数的变型unzip转换矩阵zip(*M)
# 把转换成的tuple再格式为list

def transpose(M):
    return [list(row) for row in zip(*M)]


# ## 1.5 计算矩阵乘法 AB

# In[5]:


# TODO 计算矩阵乘法 AB，如果无法相乘则返回None

# 设C=AB，首先判断AB矩阵是否可以相乘
# 根据矩阵乘法的公式算出C

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


# ## 1.6 测试你的函数是否实现正确

# **提示：** 你可以用`from pprint import pprint`来更漂亮的打印数据，详见[用法示例](http://cn-static.udacity.com/mlnd/images/pprint.png)和[文档说明](https://docs.python.org/2/library/pprint.html#pprint.pprint)。

# In[6]:


#TODO 测试1.2 返回矩阵的行和列

import pprint
pp = pprint.PrettyPrinter(indent=1, width=20)
print '测试1.2 返回矩阵的行和列'
pp.pprint(B)

print '-----正确的答案应该是-----'
print '\n (3,4)'
print '我得到的答案如下：'
print(shape(B))


#TODO 测试1.3 每个元素四舍五入到特定小数数位

print '\n'
print '测试1.3 每个元素四舍五入到特定小数数位'
C = [[1.123456, 2.123456],
     [3.123456, 4.123456]]
matxRound(C)

print '-----正确的答案应该是-----' 
print '\n[[1.12345, 2.12345],'      '\n [3.12345, 4.12345]]'
print '我得到的答案如下：'

pp.pprint(C)

#TODO 测试1.4 计算矩阵的转置

print '\n'
print '测试1.4 计算矩阵的转置'

M_test = [[1,2,3],
          [4,5,6]]
M_test_t = transpose(M_test)
pp.pprint(M_test)
print '-----正确的答案应该是-----'

print '\n [[1, 4],'      '\n [2, 5],'      '\n [3, 6]]'
print '我得到的答案如下：'
pp.pprint(M_test_t)

#TODO 测试1.5 计算矩阵乘法AB，AB无法相乘

print '\n'
print '测试1.5 计算矩阵乘法AB，AB无法相乘'
A = [[1,2], 
     [2,3], 
     [1,2]]
B = [[1,2,3,5], 
     [2,3,3,5], 
     [1,2,5,1]]
print '-----正确的答案是应该返回 None -----'
print '我得到的结果是：'
pp.pprint(matxMultiply(A, B))

#TODO 测试1.5 计算矩阵乘法AB，AB可以相乘

print '\n'
print '测试1.5 计算矩阵乘法AB，AB可以相乘'
A = [[1,2,3], 
     [2,3,3], 
     [1,2,5]]
B = [[1,2,3,5], 
     [2,3,3,5], 
     [1,2,5,1]]

print '-----正确的答案应该是-----'
print '\n [[8, 14, 24, 18],'      '\n [11, 19, 30, 28],'      '\n [10, 18, 34, 20]]'

print '我得到的答案如下：'
pp.pprint(matxMultiply(A,B))


# # 2 Gaussign Jordan 消元法
# 
# ## 2.1 构造增广矩阵
# 
# $ A = \begin{bmatrix}
#     a_{11}    & a_{12} & ... & a_{1n}\\
#     a_{21}    & a_{22} & ... & a_{2n}\\
#     a_{31}    & a_{22} & ... & a_{3n}\\
#     ...    & ... & ... & ...\\
#     a_{n1}    & a_{n2} & ... & a_{nn}\\
# \end{bmatrix} , b = \begin{bmatrix}
#     b_{1}  \\
#     b_{2}  \\
#     b_{3}  \\
#     ...    \\
#     b_{n}  \\
# \end{bmatrix}$
# 
# 返回 $ Ab = \begin{bmatrix}
#     a_{11}    & a_{12} & ... & a_{1n} & b_{1}\\
#     a_{21}    & a_{22} & ... & a_{2n} & b_{2}\\
#     a_{31}    & a_{22} & ... & a_{3n} & b_{3}\\
#     ...    & ... & ... & ...& ...\\
#     a_{n1}    & a_{n2} & ... & a_{nn} & b_{n} \end{bmatrix}$

# In[7]:


# TODO 构造增广矩阵，假设A，b行数相同

# 遍历A矩阵每行并appen对应b的每行
# 利用copy.deepcopy来深拷贝矩阵A，不改变原来的矩阵
import copy
def augmentMatrix(A, b):
    A_augment = copy.deepcopy(A)
    b_augment = copy.deepcopy(b)
    for i in range(len(b_augment)):
        A_augment[i].append(b_augment[i][0])
    return A_augment


# ## 2.2 初等行变换
# - 交换两行
# - 把某行乘以一个非零常数
# - 把某行加上另一行的若干倍：

# In[8]:


# TODO r1 <---> r2
# 直接修改参数矩阵，无返回值

def swapRows(M, r1, r2):
    M[r1], M[r2] = M[r2], M[r1]
    


# TODO r1 <--- r1 * scale， scale!=0
# 直接修改参数矩阵，无返回值

def scaleRow(M, r, scale):
    if scale == 0:
        raise ValueError
    else:
        M[r] = [i * scale for i in M[r]]
    


# TODO r1 <--- r1 + r2*scale
# 直接修改参数矩阵，无返回值

def addScaledRow(M, r1, r2, scale):
    x = [i * scale for i in M[r2]]
    M[r1] = [a + b for a, b in zip(M[r1], x)]


# ## 2.3  Gaussian Jordan 消元法求解 Ax = b

# ### 提示：
# 
# 步骤1 检查A，b是否行数相同
# 
# 步骤2 构造增广矩阵Ab
# 
# 步骤3 逐列转换Ab为化简行阶梯形矩阵 [中文维基链接](https://zh.wikipedia.org/wiki/%E9%98%B6%E6%A2%AF%E5%BD%A2%E7%9F%A9%E9%98%B5#.E5.8C.96.E7.AE.80.E5.90.8E.E7.9A.84-.7Bzh-hans:.E8.A1.8C.3B_zh-hant:.E5.88.97.3B.7D-.E9.98.B6.E6.A2.AF.E5.BD.A2.E7.9F.A9.E9.98.B5)
#     
#     对于Ab的每一列（最后一列除外）
#         当前列为列c
#         寻找列c中 对角线以及对角线以下所有元素（行 c~N）的绝对值的最大值
#         如果绝对值最大值为0
#             那么A为奇异矩阵，返回None （请在问题2.4中证明该命题）
#         否则
#             使用第一个行变换，将绝对值最大值所在行交换到对角线元素所在行（行c） 
#             使用第二个行变换，将列c的对角线元素缩放为1
#             多次使用第三个行变换，将列c的其他元素消为0
#             
# 步骤4 返回Ab的最后一列
# 
# ### 注：
# 我们并没有按照常规方法先把矩阵转化为行阶梯形矩阵，再转换为化简行阶梯形矩阵，而是一步到位。如果你熟悉常规方法的话，可以思考一下两者的等价性。

# In[9]:


# TODO 实现 Gaussain Jordan 方法求解 Ax = b

""" Gaussian Jordan 方法求解 Ax = b.
    参数
        A: 方阵 
        b: 列向量
        decPts: 四舍五入位数，默认为4
        epsilon: 判读是否为0的阈值，默认 1.0e-16
        
    返回列向量 x 使得 Ax = b 
    返回None，如果 A，b 高度不同
    返回None，如果 A 为奇异矩阵
"""


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
        Ab_T = transpose(Ab)
        # 定义当前列系数绝对值的独立列表cd
        # c绝对值合并列c_abs
        # 最大绝对值合并列c_max
        cd = []
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

A = [[1, 3, 1],
     [2, 1, 1],
     [2, 2, 1]]
b = [[11],
     [8],
     [10]]

print gj_Solve(A, b)


# ## 2.4 证明下面的命题：
# 
# **如果方阵 A 可以被分为4个部分: ** 
# 
# $ A = \begin{bmatrix}
#     I    & X \\
#     Z    & Y \\
# \end{bmatrix} , \text{其中 I 为单位矩阵，Z 为全0矩阵，Y 的第一列全0}$，
# 
# **那么A为奇异矩阵。**
# 
# 提示：从多种角度都可以完成证明
# - 考虑矩阵 Y 和 矩阵 A 的秩
# - 考虑矩阵 Y 和 矩阵 A 的行列式
# - 考虑矩阵 A 的某一列是其他列的线性组合

# TODO 请使用 latex （请参照题目的 latex 写法学习）
# 
# TODO 证明：

# In[10]:


get_ipython().run_cell_magic(u'latex', u'', u'\n$$\\text{1: \u56e0\u4e3a\u77e9\u9635\u7684\u8f6c\u7f6e\u4e0d\u6539\u53d8\u884c\u5217\u5f0f\uff0c\u5219}$$\n\n$$ At = \\begin{bmatrix}\nI    & Z \\\\\nX    & Y \\\\\n\\end{bmatrix} $$\n\n$$ \\text{2: \u53c2\u8003\u884c\u5217\u5f0f\u7684\u5c55\u5f00\u89c4\u5219}$$\n$$ \\text{ https://www.mathsisfun.com/algebra/matrix-determinant.html } $$\n$$ \\text{ \u5c55\u5f00\u8f6c\u7f6e\u540eA\u7684\u884c\u5217\u5f0f\uff1a} $$\n\n$$ \\text{3: |At| = |I|*|Y| - |Z|*|X| } $$\n\n$$ \\text{4: \u56e0\u4e3a I \u4e3a\u5355\u4f4d\u77e9\u9635\uff0c|I| = 1*1 - 0*0 = 1} $$\n\n$$ \\text{5: \u56e0\u4e3a\u5df2\u77e5Z\u4e3a\u5168 0 \u77e9\u9635\uff0c\u4e14\u77e9\u9635\u7684\u8f6c\u7f6e\u4e0d\u6539\u53d8\u884c\u5217\u5f0f\uff0c\u7efc\u4e0a\u63a8\u51fa} $$\n\n$$ \\text{|A|=|At|=1*|Y| - 0*|X|=|Y|} $$\n\n$$ \\text{6: \u6839\u636e\u63a8\u8bba\uff0c\u82e5\u884c\u5217\u5f0f\u7684\u67d0\u884c\u5168\u4e3a0\uff0c\u5219\u884c\u5217\u5f0f\u7b49\u4e8e0\uff0c\u6545  |Y| = 0 } $$\n\n$$ \\text{7: \u63a8\u51fa |A| = |Y| = 0} $$\n\n$$ \\text{8: \u6839\u636e\u5b9a\u4e49 \u77e9\u9635\u884c\u5217\u5f0f\u4e3a0\u4e0e\u77e9\u9635\u4e3a\u5947\u5f02\u77e9\u9635\u7b49\u4ef7,\u6545 A \u4e3a\u5947\u5f02\u77e9\u9635} $$')


# ## 2.5 测试 gj_Solve() 实现是否正确

# In[11]:


# TODO 构造 矩阵A，列向量b，其中 A 为奇异矩阵
# TODO 构造 矩阵A，列向量b，其中 A 为非奇异矩阵
# TODO 求解 x 使得 Ax = b
# TODO 计算 Ax
# TODO 比较 Ax 与 b


# In[12]:


print '构造 矩阵A，列向量b，其中 A 为奇异矩阵:'
A = [[1,0,2],[0,1,3],[0,3,0]]
b = [[3],[2],[3]]
x = gj_Solve(A, b)
print x


# In[13]:


print '构造 矩阵A，列向量b，其中 A 为非奇异矩阵'

A = [[1, 3, 1],
     [2, 1, 1],
     [2, 2, 1]]
b = [[11],
     [8],
     [10]]

print '求解 x 使得 Ax = b 计算 Ax'
x = gj_Solve(A, b)
print 'x:'
print x

A = [[1, 3, 1],
     [2, 1, 1],
     [2, 2, 1]]
b = [[11],
     [8],
     [10]]

Ax = matxMultiply(A, x)
    
print 'Ax:'
print Ax

print '比较 Ax 与 b:'
if Ax == b:
    print '测试成功！Ax等于b！'


# # 3 线性回归: 
# 
# ## 3.1 计算损失函数相对于参数的导数 (两个3.1 选做其一)
# 
# 我们定义损失函数 E ：
# $$
# E = \sum_{i=1}^{n}{(y_i - mx_i - b)^2}
# $$
# 
# 证明：
# $$
# \frac{\partial E}{\partial m} = \sum_{i=1}^{n}{-2x_i(y_i - mx_i - b)}
# $$
# 
# $$
# \frac{\partial E}{\partial b} = \sum_{i=1}^{n}{-2(y_i - mx_i - b)}
# $$
# 
# $$
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix} = 2X^TXh - 2X^TY
# $$
# 
# $$ 
# \text{其中 }
# Y =  \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n
# \end{bmatrix}
# ,
# X =  \begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 \\
# \end{bmatrix},
# h =  \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix}
# $$

# TODO 请使用 latex （参照题目的 latex写法学习）
# 
# TODO 证明：

# In[14]:


get_ipython().run_cell_magic(u'latex', u'', u'\n$$\\\\$$\n\n$$\n\\text{\u53d8\u6362\u540e\u7684 }X^T\\text{\u4e3a\uff1a }\n$$\n\n$$\nX^T =  \\begin{bmatrix}\nx_1 & x_2 & ... & x_n\\\\\n1 & 1 & ... & 1\\\\\n\\end{bmatrix}\n$$\n\n$$\\\\$$\n\n$$ \n\\text{\u5df2\u77e5\uff1a }\nY =  \\begin{bmatrix}\ny_1 \\\\\ny_2 \\\\\n... \\\\\ny_n\n\\end{bmatrix}\n,\nX =  \\begin{bmatrix}\nx_1 & 1 \\\\\nx_2 & 1\\\\\n... & ...\\\\\nx_n & 1 \\\\\n\\end{bmatrix},\nh =  \\begin{bmatrix}\nm \\\\\nb \\\\\n\\end{bmatrix}\n$$\n\n$$\n\\text{\u63a8\u5bfc\u51fa }\\text{\uff1a }\n$$\n\n$$\n2X^TXh - 2X^TY = -2X^T(Y-Xh)\n$$\n\n\n$$\\\\$$\n\n$$\nY-Xh = \\begin{bmatrix}\ny_1 - mx_1 - b \\\\\ny_2 - mx_2 - b \\\\\n... \\\\\ny_n - mx_n - b \\\\\n\\end{bmatrix}\n$$\n\n\n\n\n$$\n\\text{\u6240\u4ee5 }\\text{\uff1a }\n$$\n\n$$\n2X^TXh - 2X^TY = -2X^T\\begin{bmatrix}\ny_1 - mx_1 - b \\\\\ny_2 - mx_2 - b \\\\\n... \\\\\ny_n - mx_n - b \\\\\n\\end{bmatrix} = \\begin{bmatrix}\n\\sum_{i=1}^{n}{-2x_i(y_i - mx_i - b)} \\\\\n\\sum_{i=1}^{n}{-2(y_i - mx_i - b)}\n\\end{bmatrix}\n$$\n\n$$\n\\text{\u7136\u540e\u5bf9\u5b9a\u4e49\u7684\u635f\u5931\u51fd\u6570(E)\u6c42m\u3001b\u7684\u504f\u5bfc }\\text{\uff1a }\n$$\n\n\n$$\n\\frac{\\partial E}{\\partial m} = \\sum_{i=1}^{n}{2(y_i - mx_i - b)\\frac{\\partial {(y_i - mx_i - b)}}{\\partial m}} $$\n$$= \\sum_{i=1}^{n}{2(y_i - mx_i - b)(-x_i)} $$\n$$= \\sum_{i=1}^{n}{-2x_i(y_i - mx_i - b)}\n$$\n\n\n$$\n\\frac{\\partial E}{\\partial b} = \\sum_{i=1}^{n}{2(y_i - mx_i - b) \\frac{\\partial {(y_i - mx_i - b)}}{\\partial b}} $$\n$$= \\sum_{i=1}^{n}{2(y_i - mx_i - b)(-1)} $$\n$$= \\sum_{i=1}^{n}{-2(y_i - mx_i - b)}\n$$\n\n\n$$\\\\$$\n$$\\\\$$\n\n$$\n\\text{\u7efc\u4e0a\u6240\u8ff0 }\\text{\uff1a }\n$$\n\n\n$$\n\\frac{\\partial E}{\\partial m} = \\sum_{i=1}^{n}{-2x_i(y_i - mx_i - b)}\n$$\n\n$$\n\\frac{\\partial E}{\\partial b} = \\sum_{i=1}^{n}{-2(y_i - mx_i - b)}\n$$\n\n\n\n$$\n\\begin{bmatrix}\n\\frac{\\partial E}{\\partial m} \\\\\n\\frac{\\partial E}{\\partial b} \n\\end{bmatrix} = 2X^TXh - 2X^TY\n$$')


# ## 3.1 计算损失函数相对于参数的导数（两个3.1 选做其一）
# 
# 证明：
# 
# $$
# E = Y^TY -2(Xh)^TY + (Xh)^TXh
# $$ 
# 
# $$
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix}  = \frac{\partial E}{\partial h} = 2X^TXh - 2X^TY
# $$
# 
# $$ 
# \text{其中 }
# Y =  \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n
# \end{bmatrix}
# ,
# X =  \begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 \\
# \end{bmatrix},
# h =  \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix}
# $$

# TODO 请使用 latex （请参照题目的 latex 写法学习）
# 
# TODO 证明：

# ## 3.2  线性回归
# 
# ### 求解方程 $X^TXh = X^TY $, 计算线性回归的最佳参数 h

# In[15]:


# TODO 实现线性回归
'''
参数：(x,y) 二元组列表
返回：m，b
'''


# 定义线性回归方程
print '定义线性回归方程'

def linearRegression(points):
    # 构建 Ax = b 的线性方程
    X = [[points[i][0], 1] for i in range(len(points))]
   
    Y = [[points[i][1]] for i in range(len(points))]

    X_T = transpose(X)
  
    A = matxMultiply(X_T, X)
    
    b = matxMultiply(X_T, Y)
   

    #m, b = (i[0] for i in gj_Solve(A, b))
    m, b = (i[0] for i in gj_Solve(A, b, decPts=4, epsilon=1.0e-16))
    return m, b


# ## 3.3 测试你的线性回归实现

# In[16]:


# TODO 构造线性函数

# 构造线性函数

m = 2
b = 3

# TODO 构造 100 个线性函数上的点，加上适当的高斯噪音

print '构造100个线性函数上的点，加上适当的高斯噪音，均值为0方差为1'

import random

x_origin = [random.uniform(-50,50) for i in range(100)]
Y_origin = [i * m + b for i in x_origin]

x_gauss = [random.gauss(0, 1) for i in range(100)]
y_gauss = [random.gauss(0, 1) for i in range(100)]

xx = [x + y for x,y in zip(x_origin, x_gauss)]
yy = [x + y for x,y in zip(Y_origin, y_gauss)]

points = [(x,y) for x,y in zip(xx, yy)]

#TODO 对这100个点进行线性回归，将线性回归得到的函数和原线性函数比较

print '对这100个点进行线性回归，将线性回归得到的函数和原线性函数比较'

m_gauss, b_gauss = linearRegression(points)

print '原始m, b '
print m, b

print '处理后的m, b'
print m_gauss, b_gauss

print '备注：此为x和y都增加高斯噪音的数据，根据需要可以不加x的高斯噪音'


# ## 4.1 单元测试
# 
# 请确保你的实现通过了以下所有单元测试。

# In[17]:


import unittest
import numpy as np

from decimal import *

class LinearRegressionTestCase(unittest.TestCase):
    """Test for linear regression project"""

    def test_shape(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.randint(low=-10,high=10,size=(r,c))
            self.assertEqual(shape(matrix.tolist()),(r,c))


    def test_matxRound(self):

        for decpts in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()
            dec_true = [[Decimal(str(round(num,decpts))) for num in row] for row in mat]

            matxRound(mat,decpts)
            dec_test = [[Decimal(str(num)) for num in row] for row in mat]

            res = Decimal('0')
            for i in range(len(mat)):
                for j in range(len(mat[0])):
                    res += dec_test[i][j].compare_total(dec_true[i][j])

            self.assertEqual(res,Decimal('0'))


    def test_transpose(self):
        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()
            t = np.array(transpose(mat))

            self.assertEqual(t.shape,(c,r))
            self.assertTrue((matrix.T == t).all())


    def test_matxMultiply(self):

        for _ in range(10):
            r,d,c = np.random.randint(low=1,high=25,size=3)
            mat1 = np.random.randint(low=-10,high=10,size=(r,d)) 
            mat2 = np.random.randint(low=-5,high=5,size=(d,c)) 
            dotProduct = np.dot(mat1,mat2)

            dp = np.array(matxMultiply(mat1,mat2))

            self.assertTrue((dotProduct == dp).all())


    def test_augmentMatrix(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            A = np.random.randint(low=-10,high=10,size=(r,c))
            b = np.random.randint(low=-10,high=10,size=(r,1))

            Ab = np.array(augmentMatrix(A.tolist(),b.tolist()))
            ab = np.hstack((A,b))

            self.assertTrue((Ab == ab).all())

    def test_swapRows(self):
        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()

            r1, r2 = np.random.randint(0,r, size = 2)
            swapRows(mat,r1,r2)

            matrix[[r1,r2]] = matrix[[r2,r1]]

            self.assertTrue((matrix == np.array(mat)).all())

    def test_scaleRow(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()

            rr = np.random.randint(0,r)
            with self.assertRaises(ValueError):
                scaleRow(mat,rr,0)

            scale = np.random.randint(low=1,high=10)
            scaleRow(mat,rr,scale)
            matrix[rr] *= scale

            self.assertTrue((matrix == np.array(mat)).all())
    
    def test_addScaleRow(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()

            r1,r2 = np.random.randint(0,r,size=2)

            scale = np.random.randint(low=1,high=10)
            addScaledRow(mat,r1,r2,scale)
            matrix[r1] += scale * matrix[r2]

            self.assertTrue((matrix == np.array(mat)).all())


    def test_gj_Solve(self):

        for _ in range(10):
            r = np.random.randint(low=3,high=10)
            A = np.random.randint(low=-10,high=10,size=(r,r))
            b = np.arange(r).reshape((r,1))

            x = gj_Solve(A.tolist(),b.tolist())
            if np.linalg.matrix_rank(A) < r:
                self.assertEqual(x,None)
            else:
                # Ax = matxMultiply(A.tolist(),x)
                Ax = np.dot(A,np.array(x))
                loss = np.mean((Ax - b)**2)
                # print Ax
                # print loss
                self.assertTrue(loss<0.1)


suite = unittest.TestLoader().loadTestsFromTestCase(LinearRegressionTestCase)
unittest.TextTestRunner(verbosity=3).run(suite)


# In[91]:




