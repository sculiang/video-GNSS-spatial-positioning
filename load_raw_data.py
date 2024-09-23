import pandas as pd
import numpy as np

"""导入数据文件"""
file_path_list = []
file_num = 1 #for citySim
# file_num = 1 #for UE4_ZUUU
for i in range(file_num):
    num = str(i+1).zfill(2)
    file_path = 'citySim/citySim/RoundaboutB1.csv'  # for new citySim
    # file_path = 'GenerateNonGaussianNoises/Non-Gaussian/RandomWalk_WN/RoundaboutB.csv'  # for new citySim
    file_path_list.append(file_path)

"""储存"""
data_list = []
for i in range(file_num):
    data_list.append(pd.read_csv(file_path_list[i], header=None, index_col=None, low_memory=False))

# %%
'转换成ndarray'
mix_coord_list = [np.array(data_list[i].loc[1:, 1:], dtype=np.float64) for i in range(file_num)]
file_num = len(mix_coord_list)




