def demo_func(x):
    x1, x2, x3 = x
    return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2
from sko.SA import SA
sa = SA(func=demo_func, x0=[1, 1, 1])
x_star, y_star = sa.fit()
print('x的值',x_star)
print('y的值',y_star)