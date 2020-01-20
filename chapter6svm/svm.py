from __future__ import print_function
import torch as t
import random
from numpy import *

x = t.Tensor(5, 3)
print(x)
print(x.size())
y = t.rand(5, 3)
print(y)
x + y
print(x)
z = t.add(x, y)
print(z)

result = t.Tensor(5, 3)
t.add(x, y, out=result)
print(result)


class smo:

    # SMO算法相关辅助中的辅助函数
    # 解析文本数据函数，提取每个样本的特征组成向量，添加到数据矩阵
    # 添加样本标签到标签向量
    @staticmethod
    def load_data_set(file):
        dataMat = []
        labelMat = []
        fr = open(file)
        for line in fr.readlines():
            line_arr: object = line.strip().split('\t')
            dataMat.append([float(line_arr[0]), float(line_arr[1])])
            # if int(lineArr[2]) == 0 :
            # labelMat.append((float(lineArr[2]) - 1))
            # else:
            labelMat.append((float(line_arr[2])))
        return dataMat, labelMat

    # 2 在样本集中采取随机选择的方法选取第二个不等于第一个alphai的
    # 优化向量alphaj
    @staticmethod
    def selectJrand(i, m):
        j = i
        while (j == i):
            j = int(random.uniform(0, m))
        return j

    # 3 约束范围L<=alphaj<=H内的更新后的alphaj值
    @staticmethod
    def cliAlapha(aj, H, L):
        if aj > H:
            aj = H
        if L > aj:
            aj = L
        return aj

    @staticmethod
    def smoSimple(dataMatIn, classLabels, C, toler, maxIter):

        dataMatrix = mat(dataMatIn);
        labMat = mat(classLabels).transpose()
        b = 0;
        m, n = shape(dataMatrix)
        alpah = mat(zeros(m, 1))
        iter = 0
        while (iter < maxIter):
            pass
