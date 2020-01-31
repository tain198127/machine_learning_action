import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def retrieveTree(i):
    '''
   保存了树的测试数据
     '''
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', \
                                1: 'yes'}}}},{'no surfacing': {0: 'no', \
    1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]

def getNumLeafs(myTree):
    '''
    叶子节点
    '''
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDic = myTree[firstStr]
    for key in secondDic.keys():
        if type(secondDic[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDic[key])
        else:
            numLeafs += 1

    return numLeafs


def getTreeDepth(myTree):
    '''
    树深度
    '''
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    '''
    创建节点
    '''
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args
                            )
def plotMidText(cntrPt,parentPt,txtString):
    xMind = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
    yMind = (parentPt[1] - cntrPt[1])/2.0 +cntrPt[1]
    createPlot.ax1.text(xMind,yMind,txtString)

def plotTree(myTree, parentPt, nodeTxt):
    numleafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firststr = list(myTree.keys())[0]
    cntrpt = (plotTree.xoff + (1.0 + float(numleafs)) / 2.0 / plotTree.totalw, plotTree.yoff)
    # 计算子节点的坐标
    plotMidText(cntrpt, parentPt, nodeTxt)  # 绘制线上的文字
    plotNode(firststr, cntrpt, parentPt, decisionNode)  # 绘制节点
    seconddict = myTree[firststr]
    plotTree.yoff = plotTree.yoff - 1.0 / plotTree.totald
    # 每绘制一次图，将y的坐标减少1.0/plottree.totald，间接保证y坐标上深度的
    for key in seconddict.keys():
        if type(seconddict[key]).__name__ == 'dict':
            plotTree(seconddict[key], cntrpt, str(key))
        else:
            plotTree.xoff = plotTree.xoff + 1.0 / plotTree.totalw
            plotNode(seconddict[key], (plotTree.xoff, plotTree.yoff), cntrpt, leafNode)
            plotMidText((plotTree.xoff, plotTree.yoff), cntrpt, str(key))
    plotTree.yoff = plotTree.yoff + 1.0 / plotTree.totald


def createPlot(inTree):
    '''
    创建图形
    '''
    fig = plt.figure(1, facecolor='white')
    axprops=dict(xticks=[],yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalw = float(getNumLeafs(inTree))
    plotTree.totald = float(getTreeDepth(inTree))
    plotTree.xoff = -0.6 / plotTree.totalw;
    plotTree.yoff = 1.2;
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


# createPlot()
if __name__ == "__main__":
    myTree = retrieveTree(0)
    # myTree['no surfacing'][3]='maybe'
    leafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    createPlot(myTree)
    print(myTree)
    print(leafs)
    print(depth)