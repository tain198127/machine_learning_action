# -*- coding: utf-8 -*-
from tesstlog import Log

logger = Log.init_log(__name__, False)
from numpy import *
from chapter6svm.svm import *
import os
import sys
from chapter6svm.platt_smo import *
from mxnet import nd
import torch as tc
import colorful.colorful as cf

logger = Log.init_log(__name__, False)
base_path = os.path.dirname(os.path.abspath(__file__)) + "/.."
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_path)
def simplesmo():
    data, label = smo.load_data_set(current_path + '/testSet.txt')
    logger.debug(data)
    print(data)
    print('-------------')
    print(label)
    print('-------------')
    b, alphas = smo.smoSimple(data, label, 0.6, 0.001, 40)
    print(b)
    print('-------------')
    print(alphas[alphas>0])
def plattsmo():
    data, label = smo.load_data_set(current_path + '/testSet.txt')
    nb, nalpha = smoP(data, label, 0.6, 0.001, 40)
    print(nb)
    print(nalpha)
    ws = calcWs(nalpha, data, label)
    print(ws)
    dataMat = mat(data)
    v = dataMat[1] * mat(ws) + nb
    print(v)
def kernel():
    testRbf(k1=0.7)
def kernel_line():
    testRbf(kmtd='lin')
testmethod = {
    '0':simplesmo,
    '1':plattsmo,
    '2':kernel,
    '3':kernel_line
}
if (__name__ == '__main__'):
    cmd = input("选择要测试的函数：0为simple，1为platt，2为kernel,3为kernel_lin\n")
    method = testmethod.get(cmd)
    method()



