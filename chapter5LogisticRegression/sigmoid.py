from math import exp

class sigmoid:
    @staticmethod
    def loadDataSet():
        pass
    @staticmethod
    def sigmoid(z):
        ```
        ```
        return 1.0 / (1+exp(-z))
    @staticmethod
    def gradAscent(dataMatIn, classLabels):
        dataMatrix = mat(dataMatIn)
        labelMat = mat(classLabels).transpose()
    
