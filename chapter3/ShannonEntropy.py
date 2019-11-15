import Log
from math import log

logger = Log.init_log(__name__, False)


def create_date():
    dataSet = [
        [1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def calcShannonEnt(dateSet):
    numEntries = len(dateSet)
    logger.debug("numEntries=%s", numEntries)
    labelCounts = {}
    for featVec in dateSet:
        currentLabel = featVec[-1]
        logger.debug("currentLabel is %s",currentLabel)
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        # end if
        labelCounts[currentLabel] += 1
    # end for
    shannonEnt = 0.0
    for key in labelCounts:
        logger.debug("labelCounts key is %s",labelCounts[key])
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    #end for
    logger.debug("shannon ent is %s",shannonEnt)
    return shannonEnt


mydata, labels = create_date();
calcShannonEnt(mydata)
