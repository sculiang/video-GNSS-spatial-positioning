import torch
from data.data_process import X, Y, Z
from parameters_setting import batch_size
import numpy as np

# %%
'随机划分数据集训练集,比例为7:3'
'固定随机种子，保证每次训练和测试的随机划分是一致的'
random_state = 2
np.random.seed(random_state)
shuffled_index = np.random.permutation(len(X))
rate = 0.7 #for citySim
# rate = 0.8 # for UE4_ZUUU
split_index = int(len(X) * rate)
while split_index % batch_size != 0:
    split_index = split_index - 1
# %%
train_index = torch.Tensor(shuffled_index[:split_index]).long()
test_index = torch.Tensor(shuffled_index[split_index:]).long()
# %%

x_train = X[train_index, :, :]
y_train = Y[train_index, :, :]
z_train = Z[train_index, :, :]

x_test = X[test_index, :, :]
y_test = Y[test_index, :, :]
z_test = Z[test_index, :, :]