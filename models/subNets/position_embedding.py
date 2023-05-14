# *_*coding:utf-8 *_*
import math
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_seq_len=1024, learnable=False):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        if learnable:
            self.pe = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        else:
            pe = torch.zeros(max_seq_len, d_model)
            position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            #pe[:, 1::2] = torch.cos(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term) if d_model % 2 == 0 else torch.cos(position * div_term[:-1])
            pe = pe.unsqueeze(0) # Note: pe with size (1, seq_len, feature_dim)
            self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: with size (batch_size, seq_len, feature_dim)
        :return:
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)