"""
Author: Lee Sen.J
Date: 2024-09-11 11:51:04
LastEditors: chestNutLsj lisj24@mails.tsinghua.edu.cn
LastEditTime: 2024-10-26 21:24:57
FilePath: \Python-for-NumericalAnalysis\ch4-Iteratively-Rooting-Equations\1-Splitting-Method\Jocobi-method.py
Description: 雅可比迭代法，x^(k+1) = D^(-1) * (L+U)x^k + D^(-1) * b，其中D、L、U为A的对角元素部分、严格下三角元素部分的相反数、严格上三角元素部分的相反数。

"""

import numpy as np
import time

# 向量化实现
def Jocobi_vector(A, b):
    assert A.shape[0] == A.shape[1] == b.shape[0]
    x = np.zeros(b.shape, dtype=np.float32)
    d = np.diag(A)  # diag即可提取A的对角线元素，也可构建对角阵
    R = A - np.diag(d)  # r为余项
    # U = np.triu(R)   #如果想获取不含对角线的L和U需如此，直接np.triu()得到的是含对角线的
    # L = np.tril(R)
    # 迭代次数
    for t in range(n):
        x = (b - np.matmul(R, x)) / d
    return x


# 普通实现
def Jocobi_naive(A, b,x0,tol=1e-10,max_iter=1000):
    """
    使用雅可比迭代法求解 Ax = b
    
    参数：
        A (numpy.ndarray): 系数矩阵
        b (numpy.ndarray): 常数向量
        x0 (numpy.ndarray): 初始解向量
        tol (float): 误差容限
        max_iter (int): 最大迭代次数
    
    返回：
        x (numpy.ndarray): 近似解向量
        iter_count (int): 迭代次数
    """
    assert A.shape[0] == A.shape[1] == b.shape[0] # 矩阵A和向量b的维度必须相同
    # 初始化x
    x = np.zeros(b.shape, dtype=np.float32)
    # 迭代至不满足判停条件
    iter_count = 0
    while True:
        # 计算雅可比矩阵
    
    
    # for t in range(n):
    #     # 普通实现
    #     for i in range(x.shape[0]):
    #         val = b[i]
    #         for j in range(A.shape[1]):
    #             if j != i:
    #                 val -= A[i, j] * x[j]
    #         x[i] = val / A[i][i]
    return x


if __name__ == "__main__":
    # A一定要是主对角线占优矩阵
    # A = np.array(
    #     [
    #         [3, 1, -1],
    #         [2, 4, 1],
    #         [-1, 2, 5]
    #     ],dtype=np.float32
    # )
    A = np.eye(1000, dtype=np.float32)

    # b = np.array(
    #     [4, 1, 1],dtype=np.float32
    # )
    b = np.zeros((1000,), np.float32)

    start1 = time.time()
    x1 = Jocobi_naive(A, b)
    end1 = time.time()
    print("time: %.10f" % (end1 - start1))
    print(x1)

    start2 = time.time()
    x2 = Jocobi_vector(A, b)
    end2 = time.time()
    print("time: %.10f" % (end2 - start2))
    print(x2)
    # print(A, "\n", b)
