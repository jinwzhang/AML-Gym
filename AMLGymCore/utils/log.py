import logging
import os
from AMLGymCore.config.conf import LOG_DIR


def logger(log_name:str):

    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)
    folder = LOG_DIR + os.sep + log_name.split('.')[0]
    if not os.path.isdir(folder):
        os.mkdir(folder)
    logger = logging.getLogger()
    log_name = str(log_name)
    fh = logging.FileHandler(folder + os.sep + log_name)  # 可以向文件发送日志

    #ch = logging.StreamHandler()  # 可以向屏幕发送日志
    fm = logging.Formatter('%(asctime)s %(message)s')  # 打印格式


    fh.setFormatter(fm)
    #ch.setFormatter(fm)
    logger.setLevel(logging.WARN)
    logger.addHandler(fh)
    #logger.addHandler(ch)
   # logger.setLevel('INFO')  # 设置级别

    # logger.info('info')
    # logger.debug('debug')
    # logger.warning('warning')
    # logger.error('error')
    # logger.critical('critical')
    return logger
