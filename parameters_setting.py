"""模型参数设置"""
# %%
d_input = 6
d_model = 64 # original version
slight_d_model = 6 # slight self-attention version
d_ff = 256
slight_d_ff = 6
d_reg=2 #回归参量的维度，本文设为2（对应2d GNSS）
h = 4
slight_h = 1
# batch_size = 1024 #for citySim
# batch_size = 429 #for citySim
# batch_size_one_trj = 429 #for citySim
batch_size = 1000
batch_size_one_trj = 10
dropout = 0.3
epoch = 500
n_layers = 2
slight_n_layers = 1
final_output_length = 2
eps = 1e-6
LN = 'layernorm'
head = 'limited'
sequence_length = 3
'保存路径与模型名称'
self_atten_path = 'checkpoints/self-attention/'
slight_self_atten_path='checkpoints/slight self-attention/'
autoencoder_path = 'checkpoints/autoencoder/'
parameters = str(batch_size) + '-' + str(sequence_length) + '-' + str(d_model) + '-' \
             + str(n_layers) + '-' + str(h) + '-' + str(head) + '-' + str(d_ff) + '-' + str(dropout) + '-' + LN

