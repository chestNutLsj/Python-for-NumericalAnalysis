"""
Author: Lee Sen.J
Date: 2024-10-23 19:21:41
LastEditors: chestNutLsj lisj24@mails.tsinghua.edu.cn
LastEditTime: 2024-10-23 19:26:32
FilePath: \Python-for-NumericalAnalysis\ch3-Rooting-Equation-Group\31-Gaussian-Elimination\Inverse-Matrix.py
Description: 利用Gauss-Jordan消元法求矩阵的逆矩阵

"""

import numpy as np


def gauss_jordan_inverse(matrix):
    # 获取矩阵的形状
    n = matrix.shape[0]

    # 创建一个单位矩阵
    identity_matrix = np.eye(n)

    # 合并输入矩阵和单位矩阵
    augmented_matrix = np.hstack((matrix, identity_matrix))

    # 使用高斯-约当消元法
    for i in range(n):
        # 检查对角线上的元素是否为零
        if augmented_matrix[i, i] == 0:
            raise ValueError("Matrix is singular and cannot be inverted.")

        # 将对角线上的元素归一化
        augmented_matrix[i] = augmented_matrix[i] / augmented_matrix[i, i]

        # 对其余的行进行消元操作
        for j in range(n):
            if i != j:
                augmented_matrix[j] -= augmented_matrix[j, i] * augmented_matrix[i]

    # 分割出右边的单位矩阵部分，这就是逆矩阵
    inverse_matrix = augmented_matrix[:, n:]

    return inverse_matrix


# 示例
matrix = np.array([[2.0, 1.0], [5.0, 3.0]])
inverse_matrix = gauss_jordan_inverse(matrix)
print("逆矩阵为：\n", inverse_matrix)
