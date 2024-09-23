import os

'用于创建文件夹'


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
