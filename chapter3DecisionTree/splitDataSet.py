import Log
from math import log

logger = Log.init_log(__name__, False)
def split_data_set(dataSet, axis, value):
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis] == value:
            reductFeatVec = featVec[:axis]
            reductFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reductFeatVec)
    return retDataSet

a=[1,2,3]
b=[4,5,6]
a.append(b)
