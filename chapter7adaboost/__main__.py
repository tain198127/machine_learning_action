# -*- coding: utf-8 -*-
from tesstlog import Log

logger = Log.init_log(__name__, False)
import os
import sys
import colorful.colorful as cf
import torch
from chapter7adaboost.adaboost import *
if (__name__ == '__main__'):
    boost = adaboost()
    dataMat,labels = boost.load_simp_data()
    D = torch.ones((5,1))/5
    bestTrump, minError, bestClassEst = boost.build_strump(dataMat,labels,D)
    print(bestTrump, minError, bestClassEst)