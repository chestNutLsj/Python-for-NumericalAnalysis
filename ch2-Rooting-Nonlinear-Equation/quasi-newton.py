"""
Author: Lee Sen.J
Date: 2024-09-29 17:01:42
LastEditors: chestNutLsj lisj24@mails.tsinghua.edu.cn
LastEditTime: 2024-09-29 17:03:01
FilePath: \Python-for-NumericalAnalysis\ch2-Rooting-Nonlinear-Equation\quasi-newton.py
Description: 割线法代替牛顿法求解非线性方程的根
Copyright (c) 2024 by Lee Sen.J, All Rights Reserved.

"""

import numpy as np


def quasi_newton(f, df, x0, tol=1e-6, max_iter=100):
    """
    Quasi-Newton method for root finding.

    Parameters
    ----------
    f : function
        The function for which we want to find the root.
    df : function
        The derivative of the function.
    x0 : float
        An initial guess for the root.
    tol : float, optional
        The tolerance for convergence. The default is 1e-6.
    max_iter : int, optional
        The maximum number of iterations. The default is 100.

    Returns
    -------
    x : float
        The root of the function.
    """
    x = x0
    for i in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            break
        dfdx = df(x)
        if abs(dfdx) < 1e-12:
            raise ValueError("Derivative is zero.")
        x -= fx / dfdx
    else:
        raise ValueError("Maximum number of iterations reached.")
    return x


if __name__ == "__main__":

    def f(x):
        return x**3 - 2 * x**2 + 3 * x - 1

    def df(x):
        return 3 * x**2 - 4 * x + 3

    x0 = 1.5
    root = quasi_newton(f, df, x0)
    print(f"The root of the function is {root:.6f}")
