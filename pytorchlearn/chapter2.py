# -*- coding: utf-8 -*-
from tesstlog import Log

logger = Log.init_log(__name__, False)
import torch as tc
import colorful.colorful as cf


class basic_torch:
    @staticmethod
    def index_test():
        x = tc.tensor([[1.3967, 1.0892, 0.4369],
                       [1.6995, 2.0453, 0.6539],
                       [-0.1553, 3.7016, -0.3599],
                       [0.7536, 0.0870, 1.2274],
                       [2.5046, -0.1913, 0.4760]])
        y = x[1:]
        y += 1
        print(y)
        print(x[1:])
        cf.printc('使用pytorch的tensor的切片的时候，如果修改了切片的索引，那切片数据本身也改变了')
    @staticmethod
    def clone_test():
        x = tc.tensor([[1.3967, 1.0892, 0.4369],
                       [1.6995, 2.0453, 0.6539],
                       [-0.1553, 3.7016, -0.3599],
                       [0.7536, 0.0870, 1.2274],
                       [2.5046, -0.1913, 0.4760]])
        y = x.clone().view(15)[1:]
        y+=1
        print(y)
        print(x[1:])
    @staticmethod
    def idtest():
        x = tc.tensor([1,2])
        y = tc.tensor([3,4])
        cf.printc('x is {}'.format(x))
        cf.printc('y is {}'.format(y))
        y[:] = x+y
        cf.printc('x+y[:] is'.format(y))
        y = x+y
        cf.printc('x+y is'.format(x))