# -*- coding: utf-8 -*-
from tesstlog import Log
import torch

logger = Log.init_log(__name__, False)
from matplotlib import pyplot as plt


class adaboost:

    def __init__(self):
        figsize = (3.5, 2.5)
        plt.rcParams['figure.figsize'] = figsize

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
        transMat = dataArray.clone().detach()
        labelMatTranspose = classLabels.clone().detach().t()
        m, n = transMat.shape
        print(m, n)
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
                    weightError = torch.mm(D.T,errArr)

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
        weakClassArr = {}
