"""
Description: 二分法求根
Version: 1.0
Author: ZhangHongYu
Date: 2021-03-08 14:48:29
LastEditors: chestNutLsj lisj24@mails.tsinghua.edu.cn
LastEditTime: 2024-09-28 20:55:15
"""

import numpy as np
import math
import sympy as sp


def binary(f, a, b, tol):
    if f(a) == 0:
        return a
    if f(b) == 0:
        return b
    if f(a) * f(b) >= 0:
        raise ValueError("f(a)*f(b)<0 not satisfied!")
    while abs(b - a) / 2 > tol:
        c = (a + b) / 2  # 即使a和b是int,此处c自动转float了
        if f(c) == 0:  # c是一个解，完成
            return c
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2


if __name__ == "__main__":
    equation = input("Please enter the equation (e.g., x**2 - 4): ")
    # Parse the equation safely
    x = sp.symbols("x")
    f = sp.lambdify(x, sp.sympify(equation))

    interval_a = float(input("Please enter the interval a: "))
    interval_b = float(input("Please enter the interval b: "))
    tol = float(input("Please enter the tolerance: "))
    try:
        res = binary(f, interval_a, interval_b, tol)
        print(f"Root found: {res:.6f}")
    except Exception as e:
        print(f"Error: {e}")
