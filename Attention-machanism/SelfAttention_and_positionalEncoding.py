import math
import torch
from torch import nn
from d2l import torch as d2l
class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len):
        super().__init__()
        # 使用dropout降低模型对于位置信息的过于sensitive
        self.dropout = nn.Dropout(dropout)
        # 1是batch_size
        self.P = torch.zeros((1, max_len, num_hiddens))
        # shape一个是(max_len, 1) ，一个是(1, num_hiddens),刚好可以使用广播机制
        X = torch.arange(max_len, dtype=torch.float32).reshape
        (-1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        # 抄写公式即可，偶数col用sin，奇数用cos
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)
    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        # 防止对于位置过于敏感
        return self.dropout(X)