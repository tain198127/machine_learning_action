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

sys.path.append("..")
import d2lzh as d2l


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
        # feature也就是X
        features = tc.randn(num_examples, num_inputs,
                            dtype=tc.float32)
        # 表示 sigma(2*x[i][0]-3.4*x[i][1]+4.2)。labels也就是Y
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
        cf.printw('real w is{}, real b is {}'.format(w, b))
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
        return (y_hat - y.view(y_hat.size())) ** 2 / 2

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
            # 总误差
            total_loss = 0
            for X, y in self.data_iter(self.batch_size, features, labels):
                # 第一步，使用预测模型计算y_hat。定义一个模型
                y_hat = self.linreg(X, w, b)
                # 第二步，计算y_hat和y（预测和真实值之间）之间的误差，计算loss
                l = self.squared_loss(y_hat, y).sum()  # l是有关小批量X和y的损失

                # 第三步，梯度求导
                l.backward()  # 小批量的损失对模型参数求梯度
                # 优化 w和b值，不断的优化
                self.sgd([w, b], lr, self.batch_size)  # 使用小批量随机梯度下降迭代模型参数
                total_loss += l.sum()
                # 不要忘了梯度清零
                w.grad.data.zero_()
                b.grad.data.zero_()
            train_l = self.squared_loss(self.linreg(features, w, b), labels)
            print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))
            cf.printc('Epoch 第 %d 次，总loss值是 %f' % (epoch + 1, total_loss / len(features)))
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
            # end for
        # end for
        return output


class softmax_test:
    batch_size = 256
    # train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    num_inputs = 784
    num_outputs = 10


    # X = tc.tensor([[1, 2, 3], [4, 5, 6]])
    num_epochs, lr = 5, 0.1
    mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True,
                                                    transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True,
                                                   transform=transforms.ToTensor())
    """
    用来计算多类的概率，是多个output，一般MNIST模型。
    """



    def showPlt(self):
        X, y = self.info()
        d2l.show_fashion_mnist(X, d2l.get_fashion_mnist_labels(y))

    def info(self):
        print(type(self.mnist_train))
        print(len(self.mnist_train), len(self.mnist_test))
        feature, labels = self.mnist_train[0]
        print('shape is {}, labels is {}'.format(feature.shape, labels))
        X, y = [], []
        for i in range(10):
            X.append(self.mnist_train[i][0])
            y.append(self.mnist_train[i][1])
        return X, y

    def initModel(self):
        """生成权重"""
        W = tc.tensor(np.random.normal(0, 0.01, (self.num_inputs, self.num_outputs)), dtype=tc.float)
        b = tc.zeros(self.num_outputs, dtype=tc.float)
        W.requires_grad_(requires_grad=True)
        b.requires_grad_(requires_grad=True)
        return W,b
    def loadData(self):
        """加载算子"""
        if sys.platform.startswith('win'):
            num_workers = 0  # 0表示不用额外的进程来加速读取数据
        else:
            num_workers = 4
        train_iter = tc.utils.data.DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True,
                                              num_workers=num_workers)
        test_iter = tc.utils.data.DataLoader(self.mnist_test, batch_size=self.batch_size, shuffle=False,
                                             num_workers=num_workers)
        return train_iter, test_iter

    def softmax(self, X):
        X_exp = X.exp()
        partition = X_exp.sum(dim=1, keepdim=True)
        return X_exp / partition

    def net(self, X,W,b):
        return self.softmax(tc.mm(X.view((-1, self.num_inputs)),W) +b)

    def loss(self, y_hat, y):
        y_hat.gather(1, y.view(-1, 1))
        return - tc.log(y_hat.gather(1, y.view(-1, 1)))

    def accuracy(self, y_hat, y):
        return (y_hat.argmax(dim=1) == y).float().mean().item()

    def evaluate_accuracy(self, data_iter, W,b):
        acc_sum, n = 0.0, 0
        for X, y in data_iter:
            acc_sum += (self.net(X,W,b).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
        return acc_sum / n

    def train_softmax(self, net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
        W = params[0]
        b = params[1]
        for epoch in range(num_epochs):
            train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
            for X, y in train_iter:
                y_hat = net(X,W,b)
                l = loss(y_hat, y).sum()

                # 梯度清零
                if optimizer is not None:
                    optimizer.zero_grad()
                elif params is not None and params[0].grad is not None:
                    for param in params:
                        param.grad.data.zero_()

                l.backward()
                if optimizer is None:
                    d2l.sgd(params, lr, batch_size)
                else:
                    optimizer.step()  # “softmax回归的简洁实现”一节将用到

                train_l_sum += l.item()
                train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
                n += y.shape[0]
            test_acc = self.evaluate_accuracy(test_iter,W,b)
            print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
                  % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
        return W, b

    def test(self,test_iter,W,b):
        X, y = iter(test_iter).next()

        true_labels = d2l.get_fashion_mnist_labels(y.numpy())
        pred_labels = d2l.get_fashion_mnist_labels(self.net(X,W,b).argmax(dim=1).numpy())
        titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

        d2l.show_fashion_mnist(X[0:9], titles[0:9])