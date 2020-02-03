from tesstlog import Log

logger = Log.init_log(__name__, False)
import torch
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


class MLP:
    batch_size = 256

    num_inputs, num_outputs, num_hiddens = 784, 10, 256
    W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
    b1 = torch.zeros(num_hiddens, dtype=torch.float)
    W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
    b2 = torch.zeros(num_outputs, dtype=torch.float)
    params = [W1, b1, W2, b2]
    for param in params:
        param.requires_grad_(requires_grad=True)
    loss = torch.nn.CrossEntropyLoss()
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    @staticmethod
    def relu(X):
        return torch.max(input=X, other=torch.tensor(0.0))

    def net(self, X):
        X = X.view((-1, self.num_inputs))
        H = self.relu(torch.matmul(X, self.W1) + self.b1)
        return torch.matmul(H, self.W2) + self.b2

    def train(self):
        num_epochs, lr = 5, 100.0
        d2l.train_ch3(self.net, self.train_iter, self.test_iter, self.loss, num_epochs, self.batch_size, self.params,
                      lr)
        pass

    @staticmethod
    def xyplot(x_vals, y_vals, name):
        d2l.set_figsize(figsize=(5, 2.5))
        d2l.plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())
        d2l.plt.xlabel('x')
        d2l.plt.ylabel(name + '(x)')
        plt.show()
        plt.close()

    @staticmethod
    def testxplot(mtd=0):
        """
        H=ϕ(XWh+bh),
        O=HWo+bo,
​       其中ϕ就是激活函数{rele|sigmoid|tanh等}


        """
        x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)

        switcher = {
            0: x.relu,  # 分段函数
            1: x.sigmoid,  # 0-1之间
            2: x.tanh,  # -1到1之间
        }
        if switcher.get(mtd):
            y = switcher.get(mtd)()
            MLP.xyplot(x, y, 'relu')
            y.sum().backward()
            MLP.xyplot(x, x.grad, 'grad of relu')


class EasyMLP:
    num_inputs, num_outputs, num_hiddens = 784, 10, 256
    net = nn.Sequential(
        d2l.FlattenLayer(),
        nn.Linear(num_inputs, num_hiddens),
        nn.ReLU(),
        nn.Linear(num_hiddens, num_outputs),
    )
    for params in net.parameters():
        init.normal_(params, mean=0, std=0.01)
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    loss = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

    num_epochs = 5

    @staticmethod
    def sgd(params, lr, batch_size):
        # 为了和原书保持一致，这里除以了batch_size，但是应该是不用除的，因为一般用PyTorch计算loss时就默认已经
        # 沿batch维求了平均了。
        for param in params:
            param.data -= lr * param.grad / batch_size  # 注意这里更改param时用的param.data

    @staticmethod
    def evaluate_accuracy(data_iter, net, device=None):
        if device is None and isinstance(net, torch.nn.Module):
            # 如果没指定device就使用net的device
            device = list(net.parameters())[0].device
        acc_sum, n = 0.0, 0
        with torch.no_grad():
            for X, y in data_iter:
                if isinstance(net, torch.nn.Module):
                    net.eval()  # 评估模式, 这会关闭dropout
                    acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                    net.train()  # 改回训练模式
                else:  # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                    if ('is_training' in net.__code__.co_varnames):  # 如果有is_training这个参数
                        # 将is_training设置成False
                        acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                    else:
                        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
                n += y.shape[0]
        return acc_sum / n

    def train(self):
        # d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
        for epoch in range(self.num_epochs):
            train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
            for X, y in self.train_iter:
                y_hat = self.net(X)
                l = self.loss(y_hat, y).sum()

                # 梯度清零
                if self.optimizer is not None:
                    self.optimizer.zero_grad()
                elif self.params is not None and self.params[0].grad is not None:
                    for param in self.params:
                        param.grad.data.zero_()

                l.backward()
                if self.optimizer is None:
                    self.sgd(self.params, self.lr, self.batch_size)
                else:
                    self.optimizer.step()  # “softmax回归的简洁实现”一节将用到

                train_l_sum += l.item()
                train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
                n += y.shape[0]
            test_acc = self.evaluate_accuracy(self.test_iter, self.net)
            print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
                  % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
