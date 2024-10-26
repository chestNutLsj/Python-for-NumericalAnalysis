"""
Author: Lee Sen.J
Date: 2024-10-24 17:40:13
LastEditors: chestNutLsj lisj24@mails.tsinghua.edu.cn
LastEditTime: 2024-10-25 18:46:16
FilePath: \Python-for-NumericalAnalysis\ch3-Directly-Rooting-Equations\3-Piovting-Elimination\Complete-Pivoting.py
Description: 

"""

import numpy as np


# 1. 定义行列交换函数
def pivot_permute_row_col(A, p, q, k):
    """
    pivot_permute_row_col 完全主元选择

    Args:
        A (ndarray): 矩阵
        p (vector): 记录行交换的向量
        q (vector): 记录列交换的向量
        k (int): 主元所在的列的索引
    """
    # 确定子矩阵A[k:,k:]中的最大元素的索引
    max_index = np.unravel_index(np.argmax(np.abs(A[k:, k:])), A[k:, k:].shape)
    max_index = (max_index[0] + k, max_index[1] + k)

    # 行交换
    if max_index[0] != k:
        A[[k, max_index[0]]] = A[[max_index[0], k]]
        p[k], p[max_index[0]] = p[max_index[0]], p[k]

    # 列交换
    if max_index[1] != k:
        A[:, [k, max_index[1]]] = A[:, [max_index[1], k]]
        q[k], q[max_index[1]] = q[max_index[1]], q[k]


# 2. 定义主元选择函数
def complete_pivoting(A, tol=1e-10):
    """
    complete_pivoting 完全主元选择法

    Args:
        A (ndarray): 系数矩阵
        tol (float, optional): 误差容忍度. Defaults to 1e-10.
    Returns:
        A (ndarray): 分解后的矩阵（包含L和U）
        p,q (vector): 记录行交换和列交换的向量
    """
    n = A.shape[0]
    p = list(range(n))  # 记录行交换
    q = list(range(n))  # 记录列交换

    for k in range(n - 1):
        # 使用pivot_permute_row_col函数确定主元所在的列
        pivot_permute_row_col(A, p, q, k)

        # 检查主元是否为0
        if abs(A[k, k]) < tol:
            raise ValueError("Matrix is singular and cannot be factorized.")

        # 进行消去操作
        for i in range(k + 1, n):
            A[i, k] /= A[k, k]  # 存储L的第k列
            for j in range(k + 1, n):
                A[i, j] -= A[i, k] * A[k, j]  # 存储U的第k行

    return A, p, q


# 3. 替代函数
def forward_substitution(L, b):
    """
    forward_substitution 向前消元法，求解 Ly = b

    Args:
        L (ndarray): banded lower triangular matrix
        b (ndarray): 右端常数
    Returns:
        y (ndarray): 解向量
    """
    n = L.shape[0]
    y = np.zeros_like(b, dtype=np.float64)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    return y


def backward_substitution(U, y):
    """
    backward_substitution 向后消元法，求解 Ux = y

    Args:
        U (ndarray): banded upper triangular matrix
        y (ndarray): 解向量
    Returns:
        x (ndarray): 解向量
    """
    n = U.shape[0]
    x = np.zeros_like(y, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1 :], x[i + 1 :])) / U[i, i]
    return x


# 示例
if __name__ == "__main__":
    A = np.array([[2, 3, 1], [4, 1, -1], [3, 4, 1]], dtype=float)
    b = np.array([1, 2, 3], dtype=float)

    # 完全主元选择法
    A_decomp, p, q = complete_pivoting(A.copy())

    # 获得分解矩阵
    L = np.tril(A_decomp, -1) + np.eye(A_decomp.shape[0])
    U = np.triu(A_decomp)

    # 调整b以匹配行交换
    b_permuted = b[p]

    # 使用替代求解 Ax = b
    y = forward_substitution(L, b_permuted)
    x = backward_substitution(U, y)

    # 调整x以匹配列交换
    x_final = np.zeros_like(x)
    for i in range(len(q)):
        x_final[q[i]] = x[i]

    print("分解后的L：\n", L)
    print("分解后的U：\n", U)
    print("解向量x：\n", x_final)
