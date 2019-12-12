import Log
logger = Log.init_log(__name__, False)
import fire
from chapter4NaiveBayes.bayes import NaiveBayes

if __name__ == "__main__":
    def run():
        naive_bayes = NaiveBayes()
        listOPost, listClasses = naive_bayes.load_dataset()
        logger.info("load_dataset")
        logger.info(listOPost)
        logger.info(listClasses)
        vocablist = naive_bayes.create_vocablist(listOPost)
        logger.info("create_vocablist")
        logger.info(vocablist)
        train_mat = []  # 词语矩阵
        '''
        [
            [1,1,1,0,0,1,0],
            [1,1,0,0,1,1,0],
            [0,1,1,0,1,1,0]
        ]
        '''
        for postinDoc in listOPost:
            vector = naive_bayes.word2vector(vocablist, postinDoc)
            train_mat.append(vector)
        p0V, p1v, pAB = naive_bayes.train_nbo(train_mat, listClasses)
        logger.info(p0V)
        logger.info(p1v)
        logger.info(pAB)


    while(True):
        # hello = input('input a string:')
        fire.Fire({
            'run':run
        })


