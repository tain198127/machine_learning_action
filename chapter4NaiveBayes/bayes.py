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
        合并集合，并去重
        :param dataset:
        :return:
        '''
        vocab_set = set([])
        for document in dataset:
            vocab_set = vocab_set | set(document)
        ret_list = list(vocab_set)
        ret_list.sort()
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
        for word in input_set:# 循环文章中的每个词
            if word in vocab_list: #统计每个词的词，在全文set中是否出现
                return_vec[vocab_list.index(word)] = 1
            else:
                logger.info("the word %s is not in my vocabluary", word)
        return return_vec

    @staticmethod
    def train_nbo(train_matrix,train_category):
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
        num_train_docs = len(train_matrix)# 命中矩阵行数
        num_words = len(train_matrix[0])#第一篇文章的文字个数
        pabusive = sum(train_category)/float(num_train_docs)# 具有辱骂性语句的个数/总文章数
        p0Num=zeros(num_words)
        p1Num = zeros(num_words)
        p0Denom = 0.0
        p1Denom = 0.0
        for i in range(num_train_docs):#循环每一行命中矩阵
            if train_category[i] == 1:
                p1Num += train_matrix[i]#数组加法 最终形成[1,0,2,0,1,3]找各种格式的数组
                p1Denom += sum(train_matrix[i])#计算所有具有辱骂
            else:
                p0Num += train_matrix[i]
                p0Denom += sum(train_matrix[i])
        p1Vect = p1Num/p1Denom
        p0Vect = p0Num/p0Denom
        return p0Vect,p1Vect,pabusive
