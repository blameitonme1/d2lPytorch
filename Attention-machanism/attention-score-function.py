import torch
import math
from torch import nn
from d2l import torch as d2l
def masked_softmax(X, valid_lens):
    '''
    mask some elements when calculating softmax on X. Support 1D or 2D valid_lens
    '''
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim = -1)
    
class AdditiveAttention(nn.Module):
    '''
    Additivie attention scoring function, could be used when shape of queries and keys is different
    '''
    def __init__(self, key_size, querry_size, num_hidden, dropout) -> None:
        super().__init__(self)
        self.w_k = nn.Linear(key_size, num_hidden, bias=False)
        self.w_q = nn.Linear(querry_size, num_hidden, bias=False)
        self.w_v = nn.Linear(num_hidden, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.w_q(queries). self.w_k(keys)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bbm(self.dropout(self.attention_weights), values)
    