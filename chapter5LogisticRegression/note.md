# logistic回归
----

## what

logistic 算法是优化算法。本质上是一种概率估计

>回归：是个拟合的过程

sigmoid函数：是一种优秀的阶跃函数
 $$\sigma(z) = \frac{1}{1+e^{-z}} $$

sigmoid = $z = w_0x_0+w_1x_1+w_2x_2+... + w_nx_n$
= $z=w^tx$

----
## why

+ 优点 : 代价不高，容易理解
+ 缺点：容易欠拟合，精度不高
  

----

## when

+ 适用：数值型和标称型

----

## how

1. 采集数据
2. 转化为数值型，并归一化
3. 分析数据
4. 训练算法
5. 测试算法 [test]
6. 使用算法

sigmoid函数：
在每个特征上头乘以一个回归系数，然后把所有结果值相加，将总和放入sigmoid函数

## how much

>问题：回归系数应该多少最佳？
>


# 梯度上升

## what

是一种优化算法，梯度上升是求最大值，梯度下降是求最小值。

$$ \nabla f(x,y)= \big( { \frac{\delta f(x,y)}{\delta x} \atop \frac{\delta f(x,y)}{\delta y} }\big) $$



函数 $f(x,y)$ 必须在计算点上可微分

## how

sigmoid+梯度上升 = $w:= w+ \alpha\nabla _wf(w)$