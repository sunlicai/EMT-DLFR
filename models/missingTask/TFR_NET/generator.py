"""Generator Module 
@Args:      text_   ([batch_size, seq_len_t, d ])
            audio_  ([batch_size, seq_len_a, d ])
            vision_ ([batch_size, seq_len_v, d ]) 
@Returns:   text'   ([batch_size, seq_len_t, dt])
            audio'  ([batch_size, seq_len_a, da])
            vision' ([batch_size, seq_len_v, dv])
"""

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

class CTCModule(nn.Module):
    def __init__(self, args, modality='text'):
        super(CTCModule, self).__init__()
        if modality == 'text':
            out_dim, seq_len = args.feature_dims[0], args.seq_lens[0]
        elif modality == 'audio':
            out_dim, seq_len = args.feature_dims[1], args.seq_lens[1]
        elif modality == 'vision':
            out_dim, seq_len = args.feature_dims[2], args.seq_lens[2]

        # Use LSTM for predicting the position from A to B
        self.pred_output = nn.LSTM(seq_len, out_dim, num_layers=2, batch_first=True)
        
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        '''
        :input x: Input with shape [batch_size x seq_len x in_dim]
        '''
        pred_output, _ = self.pred_output(x.permute(0, 2, 1)) # batch_size x d x d'

        prob_pred_output = self.softmax(pred_output) # batch_size x d x d'
        
        pseudo_aligned_out = torch.bmm(x, prob_pred_output) # batch_size x out_seq_len x in_dim
        
        # pseudo_aligned_out is regarded as the aligned A (w.r.t B)
        return pseudo_aligned_out

class LinearTrans(nn.Module):
    def __init__(self, args, modality='text'):
        super(LinearTrans, self).__init__()
        if modality == 'text':
            in_dim, out_dim = args.dst_feature_dim_nheads[0] * 3, args.feature_dims[0]
        elif modality == 'audio':
            in_dim, out_dim = args.dst_feature_dim_nheads[0] * 3, args.feature_dims[1]
        elif modality == 'vision':
            in_dim, out_dim = args.dst_feature_dim_nheads[0] * 3, args.feature_dims[2]

        self.linear = nn.Linear(in_dim, out_dim)
        


    def forward(self, x):
        '''
        :input x: Input with shape [batch_size x seq_len x in_dim]
        '''
        return self.linear(x)


class Seq2Seq(nn.Module):
    def __init__(self, args, modality='text'):
        super(Seq2Seq, self).__init__()
        if modality == 'text':
            out_dim, in_dim = args.feature_dims[0], args.dst_feature_dim_nheads[0]*3
        elif modality == 'audio':
            out_dim, in_dim = args.feature_dims[1], args.dst_feature_dim_nheads[0]*3
        elif modality == 'vision':
            out_dim, in_dim = args.feature_dims[2], args.dst_feature_dim_nheads[0]*3

        self.decoder = nn.LSTM(in_dim, out_dim, num_layers=2, batch_first=True)

    def forward(self, x):
        return self.decoder(x)

MODULE_MAP = {
    'ctc': CTCModule, 
    'linear': LinearTrans,
}

class Generator(nn.Module):
    def __init__(self, args, modality='text'):
        super(Generator, self).__init__()

        select_model = MODULE_MAP[args.generatorModule]

        self.Model = select_model(args, modality=modality)

    def forward(self, x):
        return self.Model(x)
