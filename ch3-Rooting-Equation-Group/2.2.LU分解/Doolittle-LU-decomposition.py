"""
Author: Lee Sen.J
Date: 2024-09-11 11:51:04
LastEditors: chestNutLsj lisj24@mails.tsinghua.edu.cn
LastEditTime: 2024-10-07 16:46:47
Description: Doolittle's LU decomposition and back substitution for solving linear equations.
"""

import numpy as np
from copy import deepcopy

eps = 1e-6


# 消去步骤
def LU_decomposition(
    A,
):
    """
    Perform LU decomposition of matrix A into L and U where A = LU.
    L is a lower triangular matrix and U is an upper triangular matrix.

    Parameters:
    A (ndarray): A square matrix to decompose.

    Returns:
    L (ndarray): Lower triangular matrix.
    U (ndarray): Upper triangular matrix.
    """
    assert A.shape[0] == A.shape[1], "Matrix must be square!"
    U = deepcopy(A)  # python中变量是传拷贝，数组等对象是传引用
    L = np.zeros(A.shape, dtype=np.float32)
    for j in range(U.shape[1]):  # 消去第j列的数
        # abs(U[j ,j])为要消去的主元
        if abs(U[j, j]) < eps:
            raise ValueError("zero pivot encountered!")  # 无法解决零主元问题
            return
        L[j, j] = 1
        # 消去主对角线以下的元素A[i, j]
        for i in range(j + 1, U.shape[0]):
            mult_coeff = U[i, j] / U[j, j]
            L[i, j] = mult_coeff
            # 对这A中这一行都进行更新
            for k in range(j, U.shape[1]):
                U[i, k] = U[i, k] - mult_coeff * U[j, k]

    return L, U


# 常规的上三角进行回代(此例中对角线不为0)
def gaussion_putback_U(A, b):
    """
    Perform back substitution for upper triangular matrix A and solve for x in Ax = b.

    Parameters:
    A (ndarray): Upper triangular matrix.
    b (ndarray): Right-hand side vector.

    Returns:
    x (ndarray): Solution vector.
    """
    x = np.zeros((A.shape[0], 1), dtype=np.float32)
    for i in reversed(range(A.shape[0])):  # 算出第i个未知数
        for j in range(i + 1, A.shape[1]):
            b[i] = b[i] - A[i, j] * x[j]
        x[i] = b[i] / A[i, i]
    return x


# 下三角进行回代(此例中对角线不为0)
def gaussion_putback_L(A, b):
    """
    Perform forward substitution for lower triangular matrix A and solve for y in Ly = b.

    Parameters:
    A (ndarray): Lower triangular matrix.
    b (ndarray): Right-hand side vector.

    Returns:
    y (ndarray): Intermediate vector.
    """
    x = np.zeros((A.shape[0], 1), dtype=np.float32)
    for i in range(A.shape[0]):  # 算出第i个未知数
        for j in range(i):
            b[i] = b[i] - A[i, j] * x[j]
            # 如果b矩阵初始化时是整形，3-6.99999976 = ceil(-3.99999) = -3，
            # 直接就向上取整(截断)约成整数了
            # if i == A.shape[0] - 1:
            #     print(A[i, j], "----", x[j], "----", A[i, j]*x[j])
            #     print(b[i])
        x[i] = b[i] / A[i, i]
    return x


def LU_putback(L, U, b):
    """
    Solve the equation Ax = b using LU decomposition.

    Parameters:
    L (ndarray): Lower triangular matrix.
    U (ndarray): Upper triangular matrix.
    b (ndarray): Right-hand side vector.

    Returns:
    x (ndarray): Solution vector.
    """
    # Ax = b => LUx = b ，令Ux = Y
    # 解 LY = b
    Y = gaussion_putback_L(L, b)  # 上三角回代
    print("Y = Ux = ", Y)
    # 再解 Ux = Y
    x = gaussion_putback_U(U, Y)  # 下三角回代
    return x


if __name__ == "__main__":
    A = input("请输入A矩阵（以列表形式输入，如[[1,2,-1],[2,1,-2],[-3,1,1]]）：")
    A = np.array(eval(A), dtype=np.float32)
    b = input("请输入b矩阵（以列表形式输入，如[[1],[2],[3]]）：")
    b = np.array(
        eval(b), dtype=np.float32
    )  # 注意，此处必须是浮点型，否则整型的话会出现精度问题

    # 判断是否能进行Doolittle分解
    if np.linalg.det(A) == 0:
        print("矩阵A为奇异矩阵，无法进行LU分解！")
        exit()

    # 单纯的LU分解过程不会对b有影响
    # 即消元与回代分离
    # 进行LU分解
    L, U = LU_decomposition(A)
    print("L = ", L)
    print("U = ", U)
    # 回代得到 x
    x = LU_putback(L, U, b)
    print("x = ", x)
