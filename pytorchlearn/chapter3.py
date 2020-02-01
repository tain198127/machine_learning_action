# -*- coding: utf-8 -*-
from tesstlog import Log

logger = Log.init_log(__name__, False)
import torch as tc
import numpy as np
import torchvision
import torchvision.transforms as transforms
import time
import sys
from matplotlib import pyplot as plt
import colorful.colorful as cf
import random
import torch.utils.data as Data
import torch.nn as nn
from torch.nn import init
import torch.optim as optim


class liner_test:
    batch_size = 10

    def __init__(self):
        figsize = (3.5, 2.5)
        plt.rcParams['figure.figsize'] = figsize

    def generate_dataset(self):
        num_inputs = 2
        num_examples = 1000
        true_w = [2, -3.4]
        true_b = 4.2
        #feature也就是X
        features = tc.randn(num_examples, num_inputs,
                            dtype=tc.float32)
        #表示 sigma(2*x[i][0]-3.4*x[i][1]+4.2)。labels也就是Y
        labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
        # 表示加上一定的噪音
        labels += tc.tensor(np.random.normal(0, 0.01, size=labels.size()),
                            dtype=tc.float32)
        print(features[0], labels[0])
        plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
        plt.show()
        batch_size = 10
        for X, y in self.data_iter(batch_size, features, labels):
            print(X, y)
            break
        w = tc.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=tc.float32)
        b = tc.zeros(1, dtype=tc.float32)
        w.requires_grad_(requires_grad=True)
        b.requires_grad_(requires_grad=True)
        return features, labels, w, b

    def test(self):
        lr = 0.003
        num_epochs = 55
        features, labels, w, b = self.generate_dataset()
        cf.printw('real w is{}, real b is {}'.format(w,b))
        realw, realb = self.train(lr=lr, num_epochs=num_epochs, features=features, labels=labels, w=w, b=b)
        cf.printc('w is {}, b is {}; '.format(realw, realb))

    def linreg(self, X, w, b):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
        """
        定义模型,表示的就是 y=Xw+b
        """
        return tc.mm(X, w) + b

    def squared_loss(self, y_hat, y):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
        # 注意这里返回的是向量, 另外, pytorch里的MSELoss并没有除以 2
        """
        损失函数，表示的是 sigma{i=1,n}((hat)yi-yi)^2 这个数学公式。也就是运筹里面的求的那个最值
        y_hat是预测值，y是实际值，**2表示幂。即，将两者之间的差值做幂运算
        """
        return (y_hat - y.view(y_hat.size())) ** 2/2

    def sgd(self, params, lr, batch_size):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
        """
        优化算法：随机梯度下降
        params: [w,b]要进行求解的参数
        lr:learning rate：学习率 or 变化率
        batch_size:批次
        """
        for param in params:
            param.data -= lr * param.grad / batch_size  # 注意这里更改param时用的param.data

    def train(self, lr, num_epochs, features, labels, w, b):
        """
        num_epochs:对数据扫描的次数
        features:特征
        labels:分类结果
        w: 权重
        b:
        """
        for epoch in range(num_epochs):  # 训练模型一共需要num_epochs个迭代周期
            # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。X
            # 和y分别是小批量样本的特征和标签
            #总误差
            total_loss = 0
            for X, y in self.data_iter(self.batch_size, features, labels):
                #第一步，使用预测模型计算y_hat。定义一个模型
                y_hat = self.linreg(X, w, b)
                #第二步，计算y_hat和y（预测和真实值之间）之间的误差，计算loss
                l = self.squared_loss(y_hat, y).sum()  # l是有关小批量X和y的损失

                #第三步，梯度求导
                l.backward()  # 小批量的损失对模型参数求梯度
                #优化 w和b值，不断的优化
                self.sgd([w, b], lr, self.batch_size)  # 使用小批量随机梯度下降迭代模型参数
                total_loss+=l.sum()
                # 不要忘了梯度清零
                w.grad.data.zero_()
                b.grad.data.zero_()
            train_l = self.squared_loss(self.linreg(features, w, b), labels)
            print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))
            cf.printc('Epoch 第 %d 次，总loss值是 %f'%(epoch+1, total_loss/len(features)))
        return w, b

    def data_iter(self, batch_size, features, labels):
        num_examples = len(features)
        indices = list(range(num_examples))

        random.shuffle(indices)  # 样本的读取顺序是随机的
        for i in range(0, num_examples, batch_size):
            j = tc.LongTensor(indices[i: min(i + batch_size, num_examples)])  # 最后一次可能不足一个batch
            yield features.index_select(0, j), labels.index_select(0, j)


class easyRegress:
    num_inputs = 2
    num_examples = 1000
    num_epochs = 3
    true_w = [2, -3.4]
    true_b = 4.2
    batch_size = 10

    def __init__(self):
        self.num_inputs = 2
        self.num_examples = 1000
        self.true_w = [2, -3.4]
        self.true_b = 4.2
        self.batch_size = 10
        self.num_epochs = 3

    def generateData(self):
        """
        生成数据
        """
        features = tc.tensor(np.random.normal(0, 1, (self.num_examples, self.num_inputs)), dtype=tc.float)
        labels = self.true_w[0] * features[:, 0] + self.true_w[1] * features[:, 1] + self.true_b
        labels += tc.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=tc.float)
        return features, labels

    def read_Data(self):
        """
        转换数据
        """
        # 将训练数据的特征和标签组合
        features, labels = self.generateData()
        dataset = Data.TensorDataset(features, labels)
        # 随机读取小批量
        data_iter = Data.DataLoader(dataset, self.batch_size, shuffle=True)
        return data_iter

    def pre_train(self):
        """
        预训练
        """
        net = nn.Sequential()
        net.add_module('linear', nn.Linear(self.num_inputs, 1))
        init.normal_(net.linear.weight, mean=0, std=0.01)
        init.constant_(net.linear.bias, val=0)  # 也可以直接修改bias的data: net[0].bias.data.fill_(0)
        loss = nn.MSELoss()
        optimizer = optim.SGD(net.parameters(), lr=0.03)
        return net, loss, optimizer

    def train(self):
        net, loss, optimizer = self.pre_train()
        for epoch in range(1, self.num_epochs + 1):
            for X, y in self.read_Data():
                output = net(X)
                l = loss(output, y.view(-1, 1))
                optimizer.zero_grad()  # 梯度清零，等价于net.zero_grad()
                l.backward()
                optimizer.step()
            print('epoch %d, loss: %f' % (epoch, l.item()))
            #end for
        #end for
        return output

class softmax_test:
    """
    用来计算多类的概率，是多个output，一般MNIST模型。
    """
    def __init__(self):
        mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True,
                                                        transform=transforms.ToTensor())
        mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True,
                                                       transform=transforms.ToTensor())
