
#-*- coding: utf-8 -*-
from tesstlog import Log
logger = Log.init_log(__name__, False)
from numpy import *
from chapter6svm.svm import *
import os
import sys
from mxnet import nd
import torch as tc
logger = Log.init_log(__name__, False)
base_path = os.path.dirname(os.path.abspath(__file__))+"/.."
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_path)

STYLE = {
        'fore':
        {   # 前景色
            'black'    : 30,   #  黑色
            'red'      : 31,   #  红色
            'green'    : 32,   #  绿色
            'yellow'   : 33,   #  黄色
            'blue'     : 34,   #  蓝色
            'purple'   : 35,   #  紫红色
            'cyan'     : 36,   #  青蓝色
            'white'    : 37,   #  白色
        },

        'back' :
        {   # 背景
            'black'     : 40,  #  黑色
            'red'       : 41,  #  红色
            'green'     : 42,  #  绿色
            'yellow'    : 43,  #  黄色
            'blue'      : 44,  #  蓝色
            'purple'    : 45,  #  紫红色
            'cyan'      : 46,  #  青蓝色
            'white'     : 47,  #  白色
        },

        'mode' :
        {   # 显示模式
            'mormal'    : 0,   #  终端默认设置
            'bold'      : 1,   #  高亮显示
            'underline' : 4,   #  使用下划线
            'blink'     : 5,   #  闪烁
            'invert'    : 7,   #  反白显示
            'hide'      : 8,   #  不可见
        },

        'default' :
        {
            'end' : 0,
        },
}
def UseStyle(string, mode = '', fore = '', back = ''):

    mode  = '{}'.format(STYLE['mode'][mode] if STYLE['mode'].keys().__contains__(mode) else '')

    fore  = '{}'.format(STYLE['fore'][fore] if STYLE['fore'].keys().__contains__(fore) else '')

    back  = '{}' .format(STYLE['back'][back] if STYLE['back'].keys().__contains__(back) else '')

    style = ';'.join([s for s in [mode, fore, back] if s])

    style = '\033[{}m'.format(style if style else '')

    end   = '\033[{}m'.format(STYLE['default']['end'] if style else '')

    return '{}{}{}'.format(style, string, end)

def clolrful(content):
    return UseStyle(content, back='cyan', fore='yellow', mode='bold')
if(__name__ == '__main__'):
    data,label = smo.load_data_set(current_path+'/testSet.txt')
    logger.debug(data)
    print(data,label)
    x = nd.arange(12)
    y = tc.arange(12)
    print('mxnet arange is :{}'.format(x))
    print(clolrful('torch arange is :{}'.format(y)))

    xx = x.reshape((3,4))
    yy = y.reshape((3,4))
    print('mxnet reshap is{}',xx)
    print(clolrful('torch reshap is {}'.format(yy)))

    x0 = nd.zeros((2,3,4))
    y0 = tc.zeros((2,3,4))
    print('mxnet zeros is{}'.format(x0))
    print(clolrful('torch zeros is{}'.format(y0)))

    x1 = nd.ones((2, 3, 4))
    y1 = tc.ones((2, 3, 4))
    print('mxnet onew is{}'.format(x1))
    print(clolrful('torch onew is{}'.format(y1)))

    xary = nd.array([[1,2,3,4],[4,5,6,7],[7,8,9,10]])
    print('mxnet array is {}'.format(xary))

    # yary = tc.array([[1,2,3],[4,5,6],[7,8,9]])
    print(clolrful('torch no array'))

    xn = nd.random.normal(0,1, shape=(3,4))
    print('mxnet normal random is{}'.format(xn))
    yn = tc.rand(5,3)
    print(clolrful('torch normal random is{}'.format(yn)))

    x1ary = nd.ones((3,4))
    print('array x+y is {}'.format(x1ary+xary))

    print('multi x y is{}'.format(x1ary*xary))
    print('divide x y is{}'.format(x1ary/xary))
    print('x的自然数是 is{}'.format(xary.exp()))
    print('dot计算，自身的转置矩阵 是{}'.format(nd.dot(x1ary, xary.T)))

