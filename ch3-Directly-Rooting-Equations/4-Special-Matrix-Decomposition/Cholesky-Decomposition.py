"""
Author: Lee Sen.J
Date: 2024-09-11 11:51:04
LastEditors: chestNutLsj lisj24@mails.tsinghua.edu.cn
LastEditTime: 2024-10-25 17:13:39
FilePath: \Python-for-NumericalAnalysis\ch3-Directly-Rooting-Equations\4-Cholesky-Decomposition\Cholesky-Decomposition.py
Description: 将对称正定矩阵A分解为下三角矩阵L（L的转置记作R，上三角矩阵），即：A = L * L.T

"""

import numpy as np


def cholesky_decomposition(A):
    # 获取矩阵A的大小
    n = A.shape[0]

    # 初始化L矩阵
    L = np.zeros((n, n))

    # 基于直接三角分解的Cholesky分解
    for k in range(n):
        for i in range(k + 1):
            if i == k:
                sum_diagonal = sum(L[k][j] ** 2 for j in range(k))
                L[k][k] = np.sqrt(A[k][k] - sum_diagonal)
            else:
                sum_non_diagonal = sum(L[k][j] * L[i][j] for j in range(k))
                L[k][i] = (A[k][i] - sum_non_diagonal) / L[i][i]

    return L


# 示例
if __name__ == "__main__":
    # 示例矩阵
    A = np.array([[4, -2, 2], [-2, 2, -4], [2, -4, 11]])
    # 进行Cholesky分解
    L = cholesky_decomposition(A)

    print("矩阵L:\n", L)
    print("矩阵L的转置:\n", L.T)
