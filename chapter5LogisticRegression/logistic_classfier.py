# -*- coding: utf-8 -*-
from numpy import *
import os
import sys
base_path = os.path.dirname(os.path.abspath(__file__))+"/.."
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_path)

from tesstlog import Log
logger = Log.init_log(__name__, False)
from chapter5LogisticRegression.sigmoid import Sigmoid
class LogisticClassfier:
    """
    缺失值的处理：
    1. 以均值来补缺
    2. 特殊值补缺
    3. 忽略
    4. 相似样本的均值
    5. 用机器学习算法预测来补缺
    为什么，以0作为缺失值补缺，因为特征计算方法中，如果某个特征为0，则退化为weight = weight
    同时由于sigmoid(0) = 0.5，没有任何倾向性，因此可以使用0作为缺失值
    分类思路
    
    """
    @staticmethod
    def classify_vector(x, weights):
        """
        使用sigmoid分类函数
        :param x:输入值
        :param weights: 权重
        :return: 分类 0表示不是，1表示是
        """
        prob = Sigmoid.sigmoid(sum(x*weights))
        if prob > 0.5: 
            return 1.0
        else:
            return 0.0
    @staticmethod
    def loadSet():
        fr_train = open(current_path+"/horseColicTraining.txt")
        training_set = []
        training_label = []
        for line in fr_train.readlines():
            current_line = line.strip().split("\t")
            line_arr = []
            for i in range(21):
                line_arr.append(float(current_line[i]))
            training_set.append(line_arr)
            training_label.append(float(current_line[21]))
        return training_set,training_label

    @staticmethod
    def colicTest():
        fr_train = open(current_path+"/horseColicTraining.txt")
        fr_test = open(current_path+"/horseColicTest.txt")
        training_set = []
        training_label = []
        for line in fr_train.readlines():
            current_line = line.strip().split("\t")
            line_arr = []
            for i in range(21):
                line_arr.append(float(current_line[i]))
            training_set.append(line_arr)
            training_label.append(float(current_line[21]))
        train_wegight = Sigmoid.sto_grad_ascent(array(training_set), training_label, 500)
        errorCount = 0
        numTestVec = 0.0
        for line in fr_test.readlines():
            numTestVec += 1.0
            current_line = line.strip().split("\t")
            line_arr = []
            for i in range(21):
                line_arr.append(float(current_line[i]))
            if int(LogisticClassfier.classify_vector(array(line_arr),train_wegight)) != int(current_line[21]):
                errorCount +=1
        error_rate = (float(errorCount)/numTestVec)
        logger.debug("错误率为: %d", error_rate)
        return error_rate
    
    @staticmethod
    def multi_test():
        num_test = 10
        error_sum = 0.0
        for k in range(num_test):
            error_sum += LogisticClassfier.colicTest()
        logger.info("测试 %d 次，平均错误率为 %f", num_test, error_sum/float(num_test))
    
if __name__ == "__main__":
    LogisticClassfier.multi_test()

    set,label = LogisticClassfier.loadSet()

    weight = Sigmoid.sto_grad_ascent(set,label)
    # logger.info("随机梯度上升权重是 %s",weight)
    Sigmoid.plotBestFit(weight,set,label)