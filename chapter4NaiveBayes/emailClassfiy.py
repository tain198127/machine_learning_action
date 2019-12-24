import Log
from chapter4NaiveBayes.bayes import *
from numpy import *

logger = Log.init_log(__name__, False)


class emailClassfier:

    @staticmethod
    def textParse(bigString):
        '''
        对大词进行切割
        :param bigString:
        :return:
        '''
        import re
        listOfToken = re.split(r'\W*', bigString)
        return [tok.lower() for tok in listOfToken if len(tok) > 2]

    @staticmethod
    def spamTest():
        docList = []
        classList = []
        fullText = []
        for i in range(1, 26):
            wordlist = emailClassfier.textParse(open('email/spam/%d.txt', i).read())
            docList.append(wordlist)
            fullText.extend(wordlist)
            classList.append(1)
            wordlist = emailClassfier.textParse(open('email/ham/%d.txt', i).read())
            docList.append(wordlist)
            fullText.extend(wordlist)
            classList.append(0)
        vocabList = NaiveBayes.create_vocablist(docList)
        trainSet = range(50)
        testSet = []
        for i in range(10):
            randIndex = int(random.uniform(0, len(trainSet)))
            testSet.append(trainSet[randIndex])
            del (trainSet[randIndex])
        trainMat = []
        trainClasses = []
        for docIndex in trainSet:
            trainMat.append(NaiveBayes.word2vector(vocabList, docList[docIndex]))
            trainClasses.append(classList[docIndex])
        p0v, p1v, pSpam = NaiveBayes.train_nbo(array(trainMat), array(trainClasses))
        errorCount = 0
        for docIndex in testSet:
            wordVector = NaiveBayes.word2vector(vocabList, docList[docIndex])
            if NaiveBayes.classify_NB(array(wordVector), p0v, p1v, pSpam) != classList[docIndex]:
                errorCount += 1
        print("the error rate is :%d", float(errorCount) / len(testSet))
