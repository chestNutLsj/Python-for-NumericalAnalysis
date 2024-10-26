'''
Author: Lee Sen.J
Date: 2024-09-11 11:51:04
LastEditors: chestNutLsj lisj24@mails.tsinghua.edu.cn
LastEditTime: 2024-10-23 19:05:05
FilePath: \Python-for-NumericalAnalysis\ch3-Rooting-Equation-Group\31-Gaussian-Elimination\Naive-Gaussian.py
Description: 朴素法无法解决 主元为0 和 高斯消元 问题，高斯消元都考虑方非奇异阵(肯定是方的),计算复杂度为 O(n^3); 另一种算法是Gaussian-Jordan消元法,计算复杂度为 O(n^3),思路是将矩阵先化为上三角矩阵，再逆向线性变换保留主对角元素，实际计算复杂度多了50%

'''

import numpy as np

eps = 1e-6


# 消去步骤
def gaussion_elimination(
    A, b
):  # 假设A是方阵，A.shape[0] == A.shape[1], python中变量是传拷贝，数组等对象是传引用
    assert A.shape[0] == A.shape[1]
    for j in range(A.shape[1]):  # 消去第j列的数
        # abs(A[j ,j])为要消去的主元
        if abs(A[j, j]) < eps:
            raise ValueError("zero pivot encountered!")  # 无法解决零主元问题
            return
        # 消去主对角线以下的元素A[i, j]
        for i in range(j + 1, A.shape[0]):
            mult_coeff = A[i, j] / A[j, j]
            # 对这A中这一行都进行更新
            for k in range(j, A.shape[1]):
                A[i, k] = A[i, k] - mult_coeff * A[j, k]
            b[i] = (
                b[i] - mult_coeff * b[j]
            )  # 二维的b取单dim1的索引即[1]这种含单个元素的列表


def gaussion_putback(A, b):
    x = np.zeros((A.shape[0], 1))
    for i in reversed(range(A.shape[0])):  # 算出第i个未知数
        for j in range(i + 1, A.shape[1]):
            b[i] = b[i] - A[i, j] * x[j]
        x[i] = b[i] / A[i, i]
    return x


if __name__ == "__main__":
    A = np.array([[1, 2, -1], [2, 1, -2], [-3, 1, 1]])
    b = np.array([[3], [3], [-6]])
    gaussion_elimination(A, b)
    x = gaussion_putback(A, b)
    print(x)
    # print(A, "\n", b)
