import Log
logger = Log.init_log(__name__, False)
import fire
from chapter4NaiveBayes.bayes import NaiveBayes

if __name__ == "__main__":
    def navieBayes():
        navieBayes = NaiveBayes()
        navieBayes.testing_NB()
    navieBayes()
    # fire.Fire()





