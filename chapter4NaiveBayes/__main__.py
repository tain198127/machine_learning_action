from chapter4NaiveBayes.bayes import *

if __name__ == "__main__":
    naive_bayes = NaiveBayes()
    # listOPost, listClasses = naive_bayes.load_dataset()
    # logger.info("load_dataset")
    # logger.info(listOPost)
    # logger.info(listClasses)
    # vocablist = naive_bayes.create_vocablist(listOPost)
    # logger.info("create_vocablist")
    # logger.info(vocablist)
    # train_mat = []#词语矩阵
    # '''
    # [
    #     [1,1,1,0,0,1,0],
    #     [1,1,0,0,1,1,0],
    #     [0,1,1,0,1,1,0]
    # ]
    # '''
    # for postinDoc in listOPost:
    #     vector = naive_bayes.word2vector(vocablist, postinDoc)
    #     train_mat.append(vector)
    # p0V,p1v,pAB = naive_bayes.train_nbo(train_mat,listClasses)
    # logger.info(p0V)
    # logger.info(p1v)
    # logger.info(pAB)
    naive_bayes.testing_NB()
