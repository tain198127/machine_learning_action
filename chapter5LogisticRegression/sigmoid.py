# -*- coding: utf-8 -*-
from math import exp
from numpy import *
import Log

logger = Log.init_log(__name__, False)


class Sigmoid:
    @staticmethod
    def load_data_set():
        data_mat = []
        label_mat = []
        fr = open("./testSet.txt")
        for line in fr.readlines():
            line_array = line.strip().split()
            data_mat.append([1.0, float(line_array[0]), float(line_array[1])])  # 读X值，Y值
            label_mat.append(int(line_array[2]))  # 读标签
        return data_mat, label_mat

    @staticmethod
    def sigmoid(z):
        """
        进行分类的函数
        :param z:输入值
        :return:
        """
        return 1.0 / (1 + exp(-z))

    @staticmethod
    def grad_ascent(data_mat_in, class_labels):

        """
        这个是梯度上升算法
        首先把data_mat_in中的数据转换为numpy的矩阵
        class_label从行向量转换为列向量，
        设定步长和循环次数。
            每次循环中，都要计算矩阵* 权重=h
            计算h和标签之间的差值=error
            步长*误差值*权重*矩阵转置=权重
        最后得到权重结果
        :param data_mat_in: 数值型的矩阵
        :param class_labels: 标签
        :return: 回归系数
        """

        """形成numpy矩阵"""
        data_matrix = mat(data_mat_in)  # 形成numpy矩阵
        """由行向量转换成列向量"""
        label_matrix = mat(class_labels).transpose()  # 转换为列向量
        """计算矩阵的行数和列数"""
        m, n = shape(data_matrix)
        logger.info("data_matrix的shape 分别是 %d行和%d列", m, n)
        """每次的步长，也就是微分每次微多少"""
        alpha = 0.001
        """循环次数"""
        max_cycle = 500
        """ones是个n行的矩阵，每行的向量只有一个值，是1"""
        weight = ones((n, 1))
        logger.info("weight是 %s", weight)
        logger.info("data_matrix 转置矩阵是 :%s", data_matrix.transpose())
        """一下是这个算法的核心"""
        for k in range(max_cycle):
            """是矩阵运算"""
            h = Sigmoid.sigmoid(data_matrix * weight)
            """计算误差"""
            error = (label_matrix - h)
            """
            计算权重，权重*步长*矩阵转置 * 误差
            weight相当于每次计算 历史weight * 误差 *步长
            but why 为什么要转置？
            """
            weight = weight + alpha * data_matrix.transpose() * error
        return weight

    @staticmethod
    def sto_grad_ascent(data_mat_in, class_labels, num_inter = 150):
        """
        随机梯度上升算法
        :param data_mat_in:
        :param class_labels:
        :param num_inter: 随机循环次数
        :return:
        """
        m,n = shape(data_mat_in)
        step = 0.01
        weights = ones(n)
        for i in range(num_inter):
            dataIndex = range(m)
            for j in range(m):
                alpha = 4/(i+j+1.0)+step
                randIndex = int(random.uniform(0,len(dataIndex)))
                h = Sigmoid.sigmoid(sum(data_mat_in[randIndex] * weights))
                error = class_labels[randIndex] - h
                weights = weights + (error * alpha) * array(data_mat_in[randIndex])
        return weights

    @staticmethod
    def plotBestFit(wei, data_mat, label_mat):
        import matplotlib.pyplot as plt
        weights = array(wei)
        logger.info("wei is :%s, weights is :%s",wei, weights)
        data_arr = array(data_mat)
        n = shape(data_arr)[0]
        xcord1 = []
        ycord1 = []
        xcord2 = []
        ycord2 = []
        for i in range(n):
            if int(label_mat[i]) == 1:
                xcord1.append(data_arr[i, 1])
                ycord1.append(data_arr[i, 2])
            else:
                xcord2.append(data_arr[i, 1])
                ycord2.append(data_arr[i, 2])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
        ax.scatter(xcord2, ycord2, s=30, c='green')

        x = arange(-3.0, 3.0, 0.1)
        y = ((-weights[0] - weights[1] * x) / weights[2]).transpose()
        ax.plot(x, y)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()


if __name__ == "__main__":
    data_mat, label_mat = Sigmoid.load_data_set()
    logger.info("data: is %s", data_mat)
    logger.info("label is %s", label_mat)
    weight = Sigmoid.grad_ascent(data_mat, label_mat)
    logger.info(weight)
    Sigmoid.plotBestFit(weight,data_mat,label_mat)

    weight = Sigmoid.sto_grad_ascent(data_mat,label_mat)
    logger.info("随机梯度上升权重是 %s",weight)
    Sigmoid.plotBestFit(weight,data_mat,label_mat)
