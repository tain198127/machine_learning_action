# -*- coding: utf-8 -*-
from tesstlog import Log
import torch
import math
import numpy
logger = Log.init_log(__name__, False)
from matplotlib import pyplot as plt


class adaboost:

    def __init__(self):
        figsize = (3.5, 2.5)
        plt.rcParams['figure.figsize'] = figsize
    def _wrap_to_tensor(self,obj,deepcopy=True):
        if torch.is_tensor(obj):
            return obj
        else:
            return torch.tensor(obj)

    def load_simp_data(self):
        dataMatrix = torch.tensor([[1.0, 2.1], [2.0, 1.1], [1.3, 1.0], [1.0, 1.0], [2.0, 1.0]], dtype=torch.float)
        classLabels = torch.tensor([1.0, 1.0, -1.0, -1.0, 1.0], dtype=torch.float)

        plt.scatter(dataMatrix[:, 0].numpy(), dataMatrix[:, 1].numpy(), 3)
        plt.show()
        return dataMatrix, classLabels

    def strump_classify(self, dataMat, dim=0, thresval=100, threshIneq='lt'):
        retArray = torch.ones(dataMat.shape[0], 1)
        if threshIneq == 'lt':
            retArray[dataMat[:, dim] <= thresval] = -1.0
        else:
            retArray[dataMat[:, dim] > thresval] = 1.0

        return retArray

    def build_strump(self, dataArray, classLabels, D):
        transMat = self._wrap_to_tensor(dataArray)
        labelMatTranspose = self._wrap_to_tensor(classLabels).t()
        m, n = transMat.shape
        numSteps = 10.0
        bestTrump = {}
        bestClassEst = torch.zeros(m, 1)
        minError = float('inf')
        for i in range(n):
            rangeMin = transMat[:, i].min()
            rangeMax = transMat[:, i].max()
            stepSize = (rangeMax - rangeMin) / numSteps
            for j in range(-1, int(numSteps) + 1):
                for inequal in ['lt', 'gt']:
                    threshVal = rangeMin + float(j) * stepSize
                    predictedVals = self.strump_classify(transMat, i, threshVal, inequal)
                    errArr = torch.ones(m, 1)
                    errArr[predictedVals.view(1,5)[0] == labelMatTranspose] = 0
                    weightError = torch.mm(D.t(),errArr).sum()

                    if weightError < minError:
                        minError = weightError
                        bestClassEst = predictedVals.clone()
                        bestTrump['dim'] = i
                        bestTrump['thresh'] = threshVal
                        bestTrump['ineq'] = inequal
                    # end if
                # enf for
            # end for
        # end for
        return bestTrump, minError, bestClassEst

    def adaBoostTrainDS(self,dataArr, classLabels, numIt =40):
        dataMat = self._wrap_to_tensor(dataArr)
        labels = self._wrap_to_tensor(classLabels)
        m = dataMat.shape[0]
        D = torch.ones((m,1))/m
        aggClassEst = torch.zeros((m,1))
        weakClassArr = []
        for i in range(numIt):
            bestStrump, error, classEst = self.build_strump(dataMat,labels, D)
            print("D:{}".format(D.T))
            alpha = float(0.5*math.log((1-error)/max(error, 1e-16)))
            bestStrump['alpha']=alpha
            weakClassArr.append(bestStrump)
            print("classEst:{}".format(classEst.T))
            expon = (-1*labels.t())*(classEst)
            D = D*expon.exp()
            D = D/D.sum()
            aggClassEst+=alpha*classEst
            print("aggClassEst:{}".format(aggClassEst.T))
            aggError = torch.mul(aggClassEst.sign()!= labels.t(),torch.ones((m,1)))
            errorRate = aggError.sum()/m
            print("errorRate is :{}".format(errorRate.T))
            if errorRate == 0.0:break;
        return weakClassArr

