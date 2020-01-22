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
logger = Log.init_log(__name__, False)
base_path = os.path.dirname(os.path.abspath(__file__)) + "/.."
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_path)



if (__name__ == '__main__'):
    data, label = smo.load_data_set(current_path + '/testSet.txt')
    logger.debug(data)
    print(data, label)