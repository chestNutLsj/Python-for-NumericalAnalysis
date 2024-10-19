"""
Author: chestNutLsj lisj24@mails.tsinghua.edu.cn
Date: 2024-09-13 23:03:47
LastEditors: chestNutLsj lisj24@mails.tsinghua.edu.cn
LastEditTime: 2024-09-13 23:06:08
FilePath: \Python-for-NumericalAnalysis\ch1-Intro\wrong-calc.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
"""

import math


def quadratic_equation(a, b, c):
    """
    quadratic_equation 计算一元二次方程 ax^2 + bx + c = 0 的根

    _extended_summary_

    Args:
        a (int): _description_
        b (int): _description_
        c (int):
    """
    inter1 = 0 - b - sqrt(b**2 - 4 * a * c)
    inter2 = 0 - b + sqrt(b**2 - 4 * a * c)
    x1 = inter1 / (2 * a)
    x2 = inter2 / (2 * a)
    return x1, x2
