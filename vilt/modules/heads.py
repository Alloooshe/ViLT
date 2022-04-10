import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ITMHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc(x)
        return x


class MLMHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x


class MPPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        #TODO add CNN decoder
        self.reduce_dims = nn.Linear(config.hidden_size, 128)
        self.decoder =nn.ConvTranspose2d(128, 3, (32,32), stride=32)

    def forward(self, x,patch_indx):
        start = time.time()
        H,W = patch_indx
        x = self.transform(x)

        x = x[:,:-1,:]
        x= x.transpose(1,2)
        x = self.reduce_dims(x)
        B,D,T = x.shape
        # print("transformed x shape ", x.shape)
        x = x.view (B,D,H,W)

        x = self.decoder(x)
        t1 = time.time()
        print("decode time ", (t1 - start) / 1000)
        x= x.unfold(1, 3, 3).unfold(2, 32, 32).unfold(3, 32, 32)

        x = torch.flatten(x, start_dim=1, end_dim=3)
        end = time.time()
        print("head time ", (end - start)/1000 )

        return x
