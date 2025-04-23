import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_m, max_len=2048, dropout_prob=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_prob)

        position = torch.arange(max_len).unsqueeze(1)
        exponent = torch.arange(0, d_m, 2)
        divergence = torch.exp(exponent * (-math.log(10000.0) / d_m))
        pe = torch.zeros(1, max_len, d_m)
        pe[0, :, 0::2] = torch.sin(position * divergence)
        pe[0, :, 1::2] = torch.cos(position * divergence)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x.shape: N x T x D
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


