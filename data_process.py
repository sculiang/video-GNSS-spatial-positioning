# 数据处理 添加噪声 标准化 数据划分
import numpy as np
import torch
from data.load_raw_data import file_num, mix_coord_list
from parameters_setting import sequence_length
from data.sequence_generator import SequenceGenerator

# print("mix_coord_list:",mix_coord_list)

'按ID划分'
mix_coord_ID = []
for i in range(file_num):
    [r, c] = np.shape(mix_coord_list[i])
    id_num = mix_coord_list[i][0, 0]
    temp = mix_coord_list[i][0, 1:]
    for j in range(r - 1):
        if mix_coord_list[i][j + 1, 0] == id_num:
            temp = np.vstack((temp, mix_coord_list[i][j + 1, 1:]))
        else:
            mix_coord_ID.append(temp)
            temp = mix_coord_list[i][j + 1, 1:]
            id_num = mix_coord_list[i][j + 1, 0]
    mix_coord_ID.append(temp)
'列表长度'

# print("mix_coord_ID:",mix_coord_ID)

list_len = len(mix_coord_ID)

# %%
'寻找最短轨迹长度'
[min_len, tempc1] = np.shape(mix_coord_ID[0])

for i in range(list_len - 1):
    [tempr, tempc2] = np.shape(mix_coord_ID[i + 1])
    if min_len > tempr:
        min_len = tempr
# %%
'按照数据集中的最小轨迹长度求跳帧间隔'
re = (min_len - sequence_length) % 2
interval = max(1,(min_len - sequence_length - re) / 2) #for citySim
interval = min(5,interval)

# print("interval:",interval)

"""添加噪声"""
"""对标GPS信号加高斯噪声，噪声尺度在平均50m左右"""
mix_coord_noise_list = []
mix_coord_gt_list = []

for i in range(list_len):
    mix_coord_noise_4id=[]
    mix_coord_gt_4id=[]
    for j in range(len(mix_coord_ID[i])):
        mix_coord_noise_4id_pnt=[]
        mix_coord_noise_4id_pnt.append(mix_coord_ID[i][j][0])
        mix_coord_noise_4id_pnt.append(mix_coord_ID[i][j][1])
        mix_coord_noise_4id_pnt.append(mix_coord_ID[i][j][2])
        mix_coord_noise_4id_pnt.append(mix_coord_ID[i][j][3])
        mix_coord_noise_4id_pnt.append(mix_coord_ID[i][j][6])
        mix_coord_noise_4id_pnt.append(mix_coord_ID[i][j][7])
        mix_coord_noise_4id_pnt.append(mix_coord_ID[i][j][8])
        mix_coord_noise_4id_pnt.append(mix_coord_ID[i][j][9])
        mix_coord_noise_4id.append(mix_coord_noise_4id_pnt)

        mix_coord_gt_4id_pnt=[]
        mix_coord_gt_4id_pnt.append(mix_coord_ID[i][j][0])
        mix_coord_gt_4id_pnt.append(mix_coord_ID[i][j][1])
        mix_coord_gt_4id_pnt.append(mix_coord_ID[i][j][2])
        mix_coord_gt_4id_pnt.append(mix_coord_ID[i][j][3])
        mix_coord_gt_4id_pnt.append(mix_coord_ID[i][j][4])
        mix_coord_gt_4id_pnt.append(mix_coord_ID[i][j][5])
        mix_coord_gt_4id_pnt.append(mix_coord_ID[i][j][8])
        mix_coord_gt_4id_pnt.append(mix_coord_ID[i][j][9])
        mix_coord_gt_4id.append(mix_coord_gt_4id_pnt)

    mix_coord_noise_list.append(mix_coord_noise_4id) #for citySim
    mix_coord_gt_list.append(mix_coord_gt_4id)  # for citySim


# print("mix_coord_noise_list:",mix_coord_noise_list)
# %%

mix_coord_noise_list[0]=np.array(mix_coord_noise_list[0])
mix_coord_gt_list[0]=np.array(mix_coord_gt_list[0])

torch.set_default_tensor_type(torch.DoubleTensor)
'按三帧进行序列划分，然后合并数据集'
# print("size:",mix_coord_noise_list[0])
X = SequenceGenerator(mix_coord_noise_list[0], interval).conact()
Z = SequenceGenerator(mix_coord_gt_list[0], interval).conact()
# %%
'生成encoder输入 X，decoder输入以及ground truth Y'
for i in range(list_len - 1):
    mix_coord_noise_list[i + 1] = np.array(mix_coord_noise_list[i + 1])
    mix_coord_gt_list[i + 1] = np.array(mix_coord_gt_list[i + 1])

    X = torch.cat((X, SequenceGenerator(mix_coord_noise_list[i + 1], interval).conact()))
    Z = torch.cat((Z, SequenceGenerator(mix_coord_gt_list[i + 1], interval).conact()))
# %%
'真实标签只关注GPS坐标'
Y = Z[:, :, 4:6]

# print("X:",X)
# print("Y:",Y)
# print("Z:",Z)