import numpy as np


def lu_decomposition(A):
    # 获取矩阵的大小
    n = A.shape[0]

    # 初始化L和U矩阵
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    # 分解矩阵A为L和U
    for i in range(n):
        # 计算U的上三角部分
        for k in range(i, n):
            sum_U = sum(L[i][j] * U[j][k] for j in range(i))
            U[i][k] = A[i][k] - sum_U

        # 计算L的下三角部分
        for k in range(i, n):
            if i == k:
                L[i][i] = 1  # 对角线元素设为1
            else:
                sum_L = sum(L[k][j] * U[j][i] for j in range(i))
                L[k][i] = (A[k][i] - sum_L) / U[i][i]

    return L, U


# 示例矩阵A
A = np.array([[4, 3], [6, 3]], dtype=float)

# 进行LU分解
L, U = lu_decomposition(A)

print("矩阵 L：\n", L)
print("矩阵 U：\n", U)
