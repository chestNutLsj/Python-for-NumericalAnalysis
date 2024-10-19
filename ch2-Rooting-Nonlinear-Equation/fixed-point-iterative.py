"""
Author: Lee Sen.J
Date: 2024-09-11 11:51:04
LastEditors: chestNutLsj lisj24@mails.tsinghua.edu.cn
LastEditTime: 2024-09-29 14:33:17
FilePath: \Python-for-NumericalAnalysis\ch2-Rooting-Nonlinear-Equation\fixed-point-iterative.py
Description: 

Copyright (c) 2024 by ${git_name}, All Rights Reserved. 
"""

import numpy as np
import math


def fpi(g, x0, k):  # 迭代k次,包括x0在内共k+1个数
    x = np.zeros(
        k + 1,
    )
    x[0] = x0
    for i in range(1, k + 1):
        x[i] = g(x[i - 1])
    return x[k]


if __name__ == "__main__":
    equation = input("请输入方程：")
    x0 = float(input("请输入初值："))
    k = int(input("请输入迭代次数："))
    res = fpi(lambda x: eval(equation), x0, k)
    print(res)
