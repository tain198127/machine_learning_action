import Log
from math import log

logger = Log.init_log(__name__, False)


def create_date():
    dataSet = [
        [1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


# 对数据进行切分
def split_data_set(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reductFeatVec = featVec[:axis]
            reductFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reductFeatVec)
    return retDataSet


# 计算香农熵
def calcShannonEnt(dateSet):
    numEntries = len(dateSet)
    logger.debug("numEntries=%s", numEntries)
    labelCounts = {}
    for featVec in dateSet:
        currentLabel = featVec[-1]
        logger.debug("currentLabel is %s", currentLabel)
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        # end if
        labelCounts[currentLabel] += 1
    # end for
    shannonEnt = 0.0
    for key in labelCounts:
        logger.debug("type [%s], count is %s", key, labelCounts[key])
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    # end for
    logger.debug("shannon ent is %s", shannonEnt)
    return shannonEnt


# 选择香农熵最低的
def choose_best_feature_2_split(dataSet):
    num_feature = len(dataSet[0]) - 1
    base_entropy = calcShannonEnt(dataSet)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(num_feature):
        feat_list = [example[i] for example in dataSet]
        unique_value = set(feat_list)
        new_entropy = 0.0
        for value in unique_value:
            sub_data_set = split_data_set(dataSet, i, value)
            prob = len(sub_data_set) / float(len(dataSet))
            new_entropy += prob * calcShannonEnt(sub_data_set)
            logger.debug("prob: %s", prob)
            logger.debug("new_entropy: %s", new_entropy)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature



mydata, labels = create_date();
# calcShannonEnt(mydata)
# splitdata = split_data_set(mydata,0,1)
# logger.debug("splitdata is :%s",splitdata)
best = choose_best_feature_2_split(mydata)
print(best)
