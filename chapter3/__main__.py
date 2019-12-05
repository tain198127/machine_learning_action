from numpy import *
from chapter3.ShannonEntropy import *
from chapter3.treePlotter import *
if __name__ == '__main__':
    with open('lenses.txt') as fp:
        lenses = [line.strip().split('\t') for line in fp.readlines()]
        lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lens_tree = createTree(lenses, lensesLabels)
    createPlot(lens_tree)
