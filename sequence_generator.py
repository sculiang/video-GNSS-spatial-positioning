import numpy as np
import torch
from parameters_setting import sequence_length


class SequenceGenerator:
    """将输入数据连接成序列(即nlp里面的句子),序列长度暂时定为3帧"""

    def __init__(self, coordinate, interval):
        self.coordinate = coordinate
        self.interval = interval

    def conact(self):
        """每sequence_length个pixel gps混合数据组成sequence_length*6的矩阵，每隔固定帧数取一帧"""
        [r, c] = np.shape(self.coordinate)
        x = self.coordinate[0, :]
        '初始化第一个序列'
        for i in range(sequence_length - 1):
            x = np.block([[x], [self.coordinate[int((i + 1) * (self.interval + 1)), :]]])
        x = np.expand_dims(x, axis=0)
        '划分后续序列'
        for i in range(r):
            if i == r - 2 * int(self.interval + 1) - 1:
                break
            temp = self.coordinate[i + 1, :]
            for j in range(sequence_length - 1):
                temp = np.block([[temp], [self.coordinate[i + 1 + int((j + 1) * (self.interval + 1)), :]]])
            temp = np.expand_dims(temp, axis=0)
            '将划分好的序列连接起来'
            x = np.concatenate((x, temp), axis=0)
        '转换成tensor'
        x = torch.tensor(x, dtype=torch.float64)
        return x
