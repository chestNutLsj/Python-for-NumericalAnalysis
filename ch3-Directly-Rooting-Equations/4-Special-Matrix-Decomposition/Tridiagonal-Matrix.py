import numpy as np


def tridiagonal_matrix_decomposition(a, b, c, n):
    """
    根据三对角矩阵的尺寸和元素向量，利用直接三角分解进行LU分解，并返回LU分解后的矩阵L和U，L和U分别用m和d,c向量表示
    A = [[b1,c1, 0, 0, 0],
         [a2,b2,c2, 0, 0],
         [ 0,a3,b3,c3, 0],
         [ 0, 0,a4,b4,c4],
         [ 0, 0, 0,a5,b5]]

    L = [[1, 0, 0, 0, 0],
         [m2,1, 0, 0, 0],
         [0, m3,1, 0, 0],
         [0, 0,m4, 1, 0],
         [0, 0, 0,m5, 1]]

    U = [[d1,c1, 0, 0, 0],
         [0, d2,c2, 0, 0],
         [0, 0, d3,c3, 0],
         [0, 0,  0,d4,c4],
         [0, 0,  0, 0,d5]]

    Args:
        a (numpy.ndarray): 三对角矩阵的a元素向量
        b (numpy.ndarray): 三对角矩阵的b元素向量
        c (numpy.ndarray): 三对角矩阵的c元素向量
        n (int): 三对角矩阵的尺寸

    Returns:
        m (numpy.ndarray): 表示L矩阵的m元素向量
        d (numpy.ndarray): 表示U矩阵的d元素向量
        c (numpy.ndarray): 表示U矩阵的c元素向量
    """
    d = b.copy()
    m = np.zeros(n)
    c = c.copy()

    for i in range(2, n):
        m[i] = a[i] / d[i - 1]
        d[i] -= m[i] * c[i - 1]

    return m, d, c
