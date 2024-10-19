"""
Author: Lee Sen.J
Date: 2024-09-11 11:51:04
LastEditors: chestNutLsj lisj24@mails.tsinghua.edu.cn
LastEditTime: 2024-09-29 14:40:36
FilePath: \Python-for-NumericalAnalysis\ch2-Rooting-Nonlinear-Equation\newton-iterative.py
Description: 

Copyright (c) 2024 by ${git_name}, All Rights Reserved. 
"""

import numpy as np
import math

# import matplotlib.pyplot as plt
# import torch
import sympy as sp

# 定义函数
# def f(x):
#     return x**3 + x - 1    # 定义函数f(x) = x^3 + x - 1


def newton(x0, k, f):  # 迭代k次,包括x0在内共k+1个数
    x = x0
    for i in range(1, k + 1):
        fx = f(x)
        dfx = sp.diff(f(x), x)  # 计算导数
        x = x - fx / dfx  # 更新x
    return x  # 返回近似根


# # x.grad为dy/dx(假设dy为最后一个节点)
# def newton(x0, k, f):  # 迭代k次,包括x0在内共k+1个数
#     # 初始化计算图参数
#     x = torch.tensor([x0], requires_grad=True)
#     for i in range(1, k + 1):
#         # 前向传播，注意x要用新的对象，否则后面y.backgrad后会释放
#         y = f(x)
#         y.backward()  # y.grad是None
#         # 更新参数
#         with torch.no_grad():
#             x -= torch.divide(y, x.grad)
#         x.grad.zero_()  # 清空梯度，使下一轮建立新的计算图，否则因为backward释放资源下一轮再backward出错
#         # 注意x.grad不能是0，否则要出错使g(x)/x.grad变为none
#     return x.detach().numpy()[0]


if __name__ == "__main__":
    f = lambda x: x**3 + x - 1
    x0 = 1.0
    res = newton(x0, 10, f)
    print(res)
