import logging
import time
import os.path


def init_log(file_name,ispersistent_file):
    logging.basicConfig()
    logger = logging.getLogger(file_name)
    logger.setLevel(logging.INFO)  # Log等级总开关
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))


    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)  # 输出到console的log等级的开关
    # 第三步，定义handler的输出格式
    # formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    formatter = logging.Formatter("%(message)s")

    ch.setFormatter(formatter)

    # 第四步，将logger添加到handler里面
    logger.addHandler(ch)
    if ispersistent_file :
        log_path = os.path.dirname(os.path.realpath(file_name)) + '/tesstlog/'
        log_name = log_path + rq + '.tesstlog'
        logfile = log_name
        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.ERROR)  # 输出到file的log等级的开关
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    # 日志
    logger.debug('this is a logger debug message')
    return logger



