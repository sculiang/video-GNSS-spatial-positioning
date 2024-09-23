import torch.utils.data as data


class MyDataSet(data.Dataset):
    """自定义dataset"""

    def __init__(self, encoder_input, decoder_input, decoder_output):
        super(MyDataSet, self).__init__()
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.decoder_output = decoder_output

    def __len__(self):
        return self.encoder_input.shape[0]

    def __getitem__(self, index):
        return self.encoder_input[index], self.decoder_input[index], self.decoder_output[index]

class MyDataSet_Autoencoder(data.Dataset):
    """自定义dataset"""

    def __init__(self, dataset):
        super(MyDataSet_Autoencoder, self).__init__()
        self.dataset = dataset

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        return self.dataset[index]



