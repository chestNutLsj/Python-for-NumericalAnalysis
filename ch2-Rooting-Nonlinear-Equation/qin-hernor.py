"""
Author: Lee Sen.J
Date: 2024-09-29 17:33:58
LastEditors: chestNutLsj lisj24@mails.tsinghua.edu.cn
LastEditTime: 2024-09-29 17:34:04
FilePath: \Python-for-NumericalAnalysis\ch1-Intro\qin-henor.py
Description: 秦九韶算法（快速求多项式的值）/ Hernor's algorithm for polynomial evaluation

Copyright (c) 2024 by ${git_name}, All Rights Reserved. 
"""

import numpy as np


def qin_horner(poly, x):
    """
    秦九韶算法（快速求多项式的值）/ Hernor's algorithm for polynomial evaluation
    :param poly: 多项式系数列表/ List of polynomial coefficients
    :param x: 待求值点/ Evaluation point
    :return: 多项式在x处的值/ Value of the polynomial at x
    """
    n = len(poly) - 1
    if n == 0:
        return poly[0]
    elif n == 1:
        return poly[0] + poly[1] * x
    else:
        p = qin_horner(poly[1:], x)
        return poly[0] + p * x


# Example usage
if __name__ == "__main__":
    poly = [1, 2, 3, 4, 5]  # x^4 + 2x^3 + 3x^2 + 4x + 5
    x = 2
    print(qin_horner(poly, x))  # Output: 29.0
