# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sympy import *


def session2():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    c = a + b
    print("矩阵加法", c)
    d = a - b
    print("矩阵减法", d)
    e = a * b
    print("矩阵乘法", e)
    f = np.dot(a, b)
    print("矩阵点乘", f)
    g = np.sqrt(np.dot(a, a))
    print("矩阵长度", g)
    h = np.sqrt(np.sum((a - b) ** 2))
    print("欧氏距离", h)
    i = np.dot(a, b) / (np.sqrt(np.dot(a, a)) * np.sqrt(np.dot(b, b)))
    print("余弦相似度,越大越相似，也就表示越近", i)

    j = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    k = np.array([[4, 5, 6],
                  [7, 8, 9],
                  [10, 11, 12]])
    l = j + k
    print("矩阵相加", l)
    m = j - k
    print("矩阵相减", m)
    n = j * k
    print("矩阵相乘", n)
    o = np.dot(j, k)
    print("矩阵点乘,必须是行列一样才能点乘", o)


def sessionCal():
    # 这里是高数部分

    X = np.linspace(-10, 10, 1000)
    Y = 2 * X ** 2 + 19
    plt.plot(X, Y)
    plt.show()
    # sympy是自动求导工具

    X1 = Symbol('X1')
    Y1 = 2 * X1 ** 2 + 19
    print('Y1是：', Y1)
    # 对X1求导数
    Z1 = diff(Y1, X1)
    print('Y1对X1求导数:', Z1)

    X2 = Symbol('X2')
    Y2 = 3 * (X2 ** 2 - 2) ** 2 + 7
    Z2 = diff(Y2, X2)
    print('Y是', Y2)
    print('Y2对X2求导结果', Z2)

    # 求偏导数
    X3, Y3 = symbols('X3 Y3')
    Z3 = X3 ** 2 + 3 * X3 * Y3 + Y3 ** 2
    print('Z3是', Z3)
    result1 = diff(Z3, X3)
    result2 = diff(Z3, Y3)
    print('对X3求偏导是', result1)
    print('对Y3求骗到是', result2)
    # 一般用在特征工程

    # 幂级数——>用来拟合数据的，例如拟合股票波动？
    # a0X^0+a1X^1+a2X^2......anX^n,这不就是泰勒展开


def session5():
    # a = np.random.randint(1,1000,1000)
    # print(a)
    b = np.array([[1, 2], [3, 4], [5, 6]])
    print('shape是:{};size 是 {}'.format(b.shape, b.size))
    print(b)
    c = b.reshape(2, 3)
    print('shape是:{};size 是 {}'.format(c.shape, c.size))
    print(c)
    d = pd.DataFrame(data=np.array([[175, 150, 36],
                                    [172, 160, 38],
                                    [173, 170, 44]]),
                                    index=[1, 2, 3],
                            columns=['身高', '体重', '胸围'])
    print(d)
    print('========================================')

    path = '~/Desktop/生物信息.csv'
    e = pd.DataFrame(pd.read_csv(path))

    print(e)
    print('===============数据内容======================')
    print('columns:\n')
    print(e.columns)
    print('values:\n')
    print(e.values)

    print(e.loc[0:,'身高'].values)
    # plt.plot(e)
    plt.scatter(x= e.loc[0:,'身高'],y=e.loc[0:,'胸围'])
    plt.show()


session5()
