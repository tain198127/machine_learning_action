import Log
import operator
from math import log

logger = Log.init_log(__name__, False)

'''
表决
'''
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
'''
创建决策树
'''
def createTree(dataset, test_labels):
    classList = [x[-1] for x in dataset]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #
    if len(dataset[0]) == 1:
        return majorityCnt(classList)

    best_feature = choose_best_feature_2_split(dataset)
    best_labels = test_labels[best_feature]

    myTree = {best_labels: {}}
    # 此位置书上写的有误，书上为del(labels[bestFeat])
    # 相当于操作原始列表内容，导致原始列表内容发生改变
    # 按此运行程序，报错'no surfacing'is not in list
    # 以下代码已改正

    # 复制当前特征标签列表，防止改变原始列表的内容
    subLabels = test_labels[:]
    # 删除属性列表中当前分类数据集特征
    del (subLabels[best_feature])

    # 使用列表推导式生成该特征对应的列
    f_val = [x[best_feature] for x in dataset]
    uni_val = set(f_val)
    for value in uni_val:
        # 递归创建子树并返回
        myTree[best_labels][value] = createTree(split_data_set(dataset, best_feature, value), subLabels)

    return myTree
'''
创建数据
'''
def create_date():
    data_set = [
        [1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


'''
对数据进行切分
'''
def split_data_set(data_set, axis, value):
    ret_data_set = []
    for featVec in data_set:
        if featVec[axis] == value:
            reduce_feat_vec = featVec[:axis]
            reduce_feat_vec.extend(featVec[axis + 1:])
            ret_data_set.append(reduce_feat_vec)
    return ret_data_set


'''
计算香农熵
'''
def calcShannonEnt(dateSet):
    num_entries = len(dateSet)
    logger.debug("num_entries=%s", num_entries)
    label_counts = {}
    for featVec in dateSet:
        current_label = featVec[-1]
        logger.debug("current_label is %s", current_label)
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        # end if
        label_counts[current_label] += 1
    # end for
    shannon_ent = 0.0
    for key in label_counts:
        logger.debug("type [%s], count is %s", key, label_counts[key])
        prob = float(label_counts[key]) / num_entries
        shannon_ent -= prob * log(prob, 2)
    # end for
    logger.debug("shannon ent is %s", shannon_ent)
    return shannon_ent


'''
选择熵最低的
'''
def choose_best_feature_2_split(dataSet):
    # 计算有几列
    num_feature = len(dataSet[0]) - 1
    # 计算这一列的香农熵
    base_entropy = calcShannonEnt(dataSet)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(num_feature):
        # 每次循环，去除循环所在列的数据
        feat_list = [example[i] for example in dataSet]
        # 去重，形成feature
        unique_value = set(feat_list)
        new_entropy = 0.0
        for value in unique_value:
            # 根据每个feature，对dataset进行分组
            sub_data_set = split_data_set(dataSet, i, value)
            # 计算基尼不纯度
            prob = len(sub_data_set) / float(len(dataSet))
            new_entropy += prob * calcShannonEnt(sub_data_set)
            logger.debug("prob: %s", prob)
            logger.debug("new_entropy: %s", new_entropy)
        # 计算总的基尼不纯度
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


mydata, labels = create_date()
print(mydata)

# calcShannonEnt(mydata)
# splitdata = split_data_set(mydata,0,1)
# logger.debug("splitdata is :%s",splitdata)
# best = choose_best_feature_2_split(mydata)
tree = createTree(mydata, labels)
print(tree)
