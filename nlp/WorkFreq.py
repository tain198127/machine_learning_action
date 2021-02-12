import jieba
import jieba.posseg as pseg
import re
import fire
import tkinter as tk
from tkinter import ttk

# jieba.enable_paddle()# 启动paddle模式。 0.40版之后开始支持，早期版本不支持
jieba.enable_parallel(4)


def analysis(path):
    wordbeg = dict()

    f = open(path)
    line = f.readline()
    while line:

        line = f.readline()

        fil = re.compile(u'[^0-9a-zA-Z\u4e00-\u9fa5.，,。？“”]+', re.UNICODE)
        retxt = fil.sub('',line)
        words = pseg.cut(retxt,use_paddle=True)
        for word, flg in words:
            if {'v','n','nr','vn'}.__contains__(flg):
                comKey = word+","+flg
                if wordbeg.keys().__contains__(comKey):
                    wordbeg[comKey] = wordbeg.get(comKey) + 1
                else:
                    wordbeg[comKey] = 1
    f.close()
    for key in wordbeg:
        print(key, ',',wordbeg[key])


fire.Fire(analysis)
