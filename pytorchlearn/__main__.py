# -*- coding: utf-8 -*-
from tesstlog import Log

logger = Log.init_log(__name__, False)
from numpy import *
from chapter6svm.svm import *
import os
import sys
from mxnet import nd
import torch as tc
import colorful.colorful as cf
from pytorchlearn.chapter2 import *
from pytorchlearn.chapter3 import *

logger = Log.init_log(__name__, False)
base_path = os.path.dirname(os.path.abspath(__file__)) + "/.."
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_path)


def basictest():
    bt = basic_torch()
    bt.index_test()
    bt.clone_test()
    bt.idtest()


def linertest():
    lt = liner_test()
    lt.test()


def easyregress():
    er = easyRegress()
    net = er.train()
    cf.printw('easy regress net is{}'.format(net))


def softmaxtest():
    st = softmax_test()

    trainer,tester = st.loadData()
    real_W,real_b = st.initModel()
    W,b = st.train_softmax(st.net,trainer,tester,st.loss,st.num_epochs,st.batch_size,[real_W,real_b],st.lr)
    st.test(tester,W,b)

testmethod = {
    '0': ['基础函数', basictest],
    '1': ['线性回归', linertest],
    '2': ['自动版线性回归', easyregress],
    '3': ['softmax', softmaxtest]
}
if (__name__ == '__main__'):
    command = '============================\n退出请输入-1\n'
    for key in testmethod.keys():
        command += key + '是:' + testmethod.get(key)[0] + '\n'
    command += ':'
    cmd = '65535'
    while True:
        cmd = input(command).strip()
        if int(cmd) <0:
            break
        if cmd in testmethod.keys():
            method = testmethod.get(cmd)
            method[1]()
        else:
            cf.printc('输入的命令不在可选范围内')

    # x = nd.arange(12)
    # y = tc.arange(12)
    # print('mxnet arange is :{}'.format(x))
    # cf.printc('torch arange is :{}'.format(y))
    #
    # xx = x.reshape((3, 4))
    # yy = y.reshape((3, 4))
    # print('mxnet reshap is{}', xx)
    # cf.printc('torch reshap is {}'.format(yy))
    #
    # x0 = nd.zeros((2, 3, 4))
    # y0 = tc.zeros((2, 3, 4), dtype=long)
    # print('mxnet zeros is{}'.format(x0))
    # cf.printc('torch zeros is{}'.format(y0))
    #
    # x1 = nd.ones((2, 3, 4))
    # y1 = tc.ones((2, 3, 4))
    # print('mxnet onew is{}'.format(x1))
    # cf.printc('torch onew is{}'.format(y1))
    #
    # xary = nd.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]])
    # print('mxnet array is {}'.format(xary))
    #
    # # yary = tc.array([[1,2,3],[4,5,6],[7,8,9]])
    # print(cf.cololrfull('torch no array'))
    #
    # xn = nd.random.normal(0, 1, shape=(3, 4))
    # print('mxnet normal random is{}'.format(xn))
    # yn = tc.rand(5, 3)
    # print(cf.cololrfull('torch normal random is{}'.format(yn)))
    #
    # x1ary = nd.ones((3, 4))
    # print('array x+y is {}'.format(x1ary + xary))
    #
    # print('multi x y is{}'.format(x1ary * xary))
    # print('divide x y is{}'.format(x1ary / xary))
    # print('x的自然数是 is{}'.format(xary.exp()))
    # print('dot计算，自身的转置矩阵 是{}'.format(nd.dot(x1ary, xary.T)))
    #
    # e = tc.eye(10)
    # cf.printc('torch eye is {}'.format(e))
    # cf.printc('torch split is {}'.format(tc.split(e, 3)))
    # slice = e[3, :]
    # cf.printc('切片结果:{}'.format(slice))
    # slice += 1
    # cf.printc('切片进行加法处理 :{}'.format(slice))
    # cf.printc('对切片的view3进行观察{}'.format(slice.view(2, 5)))
    # cf.printc('对切片的view的克隆进行变换{}'.format(slice.view(2, 5).clone() + 1))
    # cf.printc('对切片的view重新进行观察{}'.format(slice.view(2, 5)))
    # cf.printc('对切片的view进行修改{}'.format(slice.view(2, 5) + 1))
    # cf.printc('对切片的view重新进行观察{}'.format(slice.view(2, 5)))
    # if tc.cuda.is_available():
    #     device = tc.device('cuda')
    #     cf.printc('启用cuda的效果{}'.format(device))
    #
    # print('自动求梯度')
    # xgrad = tc.ones(2, 2, requires_grad=True)
    # print(xgrad)
    # print(xgrad.grad_fn)
    # ygrad = xgrad + 2
    # print(ygrad)
    # print(ygrad.grad_fn)
    # print(xgrad.is_leaf, ygrad.is_leaf)
    # zgrad = ygrad * ygrad * 3
    # out = zgrad.mean()
    # print(zgrad, out)
    # out.backward()
    # cf.printc("xgrad自动求导结果{}".format(xgrad.grad))
    # xgrad.grad.data.zero_()
    # zzgard = ygrad * ygrad * 3
    # zzout = zzgard.mean()
    # zzout.backward()
    # cf.printc('再次自动求导{}'.format(xgrad.grad))
    # cf.printw('求导只能对标量求导，不能对矩阵(张量)求导')
    # print(
    #     '所以必要时我们要把张量通过将所有张量的元素加权求和的方式转换为标量，举个例子，假设y由自变量x计算而来，w是和y同形的张量，则y.backward(w)的含义是：先计算l = torch.sum(y * w)，则l是个标量，然后求l对自变量x的导数')
    # print('反正啦，pytorch不支持直接对张量进行求导，需要以1=sum(x,y)的方式，苟且一下')
