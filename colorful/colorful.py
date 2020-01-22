STYLE = {
    'fore':
        {  # 前景色
            'black': 30,  # 黑色
            'red': 31,  # 红色
            'green': 32,  # 绿色
            'yellow': 33,  # 黄色
            'blue': 34,  # 蓝色
            'purple': 35,  # 紫红色
            'cyan': 36,  # 青蓝色
            'white': 37,  # 白色
        },

    'back':
        {  # 背景
            'black': 40,  # 黑色
            'red': 41,  # 红色
            'green': 42,  # 绿色
            'yellow': 43,  # 黄色
            'blue': 44,  # 蓝色
            'purple': 45,  # 紫红色
            'cyan': 46,  # 青蓝色
            'white': 47,  # 白色
        },

    'mode':
        {  # 显示模式
            'mormal': 0,  # 终端默认设置
            'bold': 1,  # 高亮显示
            'underline': 4,  # 使用下划线
            'blink': 5,  # 闪烁
            'invert': 7,  # 反白显示
            'hide': 8,  # 不可见
        },

    'default':
        {
            'end': 0,
        },
}


def UseStyle(string, mode='', fore='', back=''):
    mode = '{}'.format(STYLE['mode'][mode] if STYLE['mode'].keys().__contains__(mode) else '')

    fore = '{}'.format(STYLE['fore'][fore] if STYLE['fore'].keys().__contains__(fore) else '')

    back = '{}'.format(STYLE['back'][back] if STYLE['back'].keys().__contains__(back) else '')

    style = ';'.join([s for s in [mode, fore, back] if s])

    style = '\033[{}m'.format(style if style else '')

    end = '\033[{}m'.format(STYLE['default']['end'] if style else '')

    return '{}{}{}'.format(style, string, end)


def cololrfull(content):
    return UseStyle(content, back='cyan', fore='yellow', mode='bold')


def color_warn(content):
    return UseStyle(content, back='white', fore='red', mode='bold')

def printc(*values, sep=' ', end='\n', file=None):
    print(cololrfull(values), sep, end, file)


def printw(*values, sep=' ', end='\n', file=None):
    print(color_warn(values), sep, end, file)
