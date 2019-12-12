import Log
from numpy import *

logger = Log.init_log(__name__, False)


class NaiveBayes:
    @staticmethod
    def load_dataset():
        '''
        postingList: 进行词条切分后的文档集合
        classVec:类别标签
        使用伯努利模型的贝叶斯分类器只考虑单词出现与否（0，1）
        :returns:posting_list 原文
        :returns class_vec分类
        '''
        posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
        class_vec = [0, 1, 0, 1, 0, 1]  # 1代表侮辱性文字，0代表正常言论。表示每句话中，是否有侮辱文字，这个过程还是由人来做的
        return posting_list, class_vec

    @staticmethod
    def create_vocablist(dataset):
        '''
        合并集合，并去重，形成去重后的set
        :param dataset:
        :return:
        '''
        vocab_set = set([])
        for document in dataset:
            vocab_set = vocab_set | set(document)
        ret_list = list(vocab_set)
        # ret_list.sort()
        return ret_list

    @staticmethod
    def word2vector(vocab_list, input_set):
        '''
        文字转向量
        :param vocab_list:所有文章形成的词的set
        :param input_set:每段文章
        :return: 文章中某个词是否在全文set中出现
        例如 vocab_list = ["张三","打了","王五","爱上"]；input_set=["张三","爱上","王五"]。那么返回值就是
        [1，0，1，1]
        '''
        return_vec = [0] * len(vocab_list)  # 创建一个vocablist长度的向量
        for word in input_set:  # 循环文章中的每个词
            if word in vocab_list:  # 统计每个词的词，在全文set中是否出现
                return_vec[vocab_list.index(word)] = 1
            else:
                logger.info("the word %s is not in my vocabluary", word)
        return return_vec

    @staticmethod
    def train_nbo(train_matrix, train_category):
        '''
        训练朴素贝叶斯
        :param train_matrix: 词语命中矩阵，例如
        [
            [1,1,1,0,0,1,0],
            [1,1,0,0,1,1,0],
            [0,1,1,0,1,1,0]
        ]
        :param train_category: 分类标签，[1,0,1,1,0]，1表示辱骂，0表示没有辱骂
        :return:
        '''
        num_train_docs = len(train_matrix)  # 命中矩阵行数
        num_words = len(train_matrix[0])  # 第一篇文章的文字个数
        pabusive = sum(train_category) / float(num_train_docs)  # 具有辱骂性语句的个数/总文章数
        p0Num = ones(num_words)
        p1Num = ones(num_words)
        p0Denom = 2.0
        p1Denom = 2.0
        for i in range(num_train_docs):  # 循环每一行命中矩阵
            if train_category[i] == 1:
                p1Num += train_matrix[i]  # 数组加法 最终形成[1,0,2,0,1,3]找各种格式的数组
                p1Denom += sum(train_matrix[i])  # 计算所有具有辱骂
            else:
                p0Num += train_matrix[i]
                p0Denom += sum(train_matrix[i])
        '''
         这里为什么要使用对数，因为如果p1Num是个很小的值，就会造成vect为0.
         使用对数可以解决这个问题，自然对数的特性会让其导数和原值不变，即，变化率不变。但是数据更加容易处理
         虽然最终的结果不一样，但是不影响
        '''
        p1Vect = log(p1Num / p1Denom)
        p0Vect = log(p0Num / p0Denom)
        return p0Vect, p1Vect, pabusive

    @staticmethod
    def classify_NB(vec2_classify, p0_vec, p1_vec, p_class1):
        '''
        进行分类
        :param vec2_classify: 文章命中矩阵
        :param p0_vec: 命中概率
        :param p1_vec: 未命中概率
        :param p_class1:具有辱骂性语句的个数/总文章数
        :return:
        '''
        p1 = sum(vec2_classify * p1_vec) + log(p_class1)
        p0 = sum(vec2_classify * p0_vec) + log(1.0 - p_class1)
        if p1 > p0:
            return 1
        else:
            return 0

    @staticmethod
    def testing_NB():
        listOPost, listClasses = NaiveBayes.load_dataset()
        vocablist = NaiveBayes.create_vocablist(listOPost)
        train_matrix = []
        for postin_doc in listOPost:
            train_matrix.append(NaiveBayes.word2vector(vocablist,postin_doc))
        p0v,p1v,pab = NaiveBayes.train_nbo(train_matrix,listClasses)
        testEntity = ['love','my','dalmation']
        doc_matrix = array(NaiveBayes.word2vector(vocablist,testEntity))

        print(NaiveBayes.classify_NB(doc_matrix,p0v,p1v,pab))

        testEntity1 = ['stupid','garbage']
        doc_matrix1 = array(NaiveBayes.word2vector(vocablist, testEntity1))
        print(NaiveBayes.classify_NB(doc_matrix1, p0v, p1v, pab))