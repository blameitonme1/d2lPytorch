import torch
from torch import nn
from d2l import torch as d2l
import collections
import math
class Seq2SeqEncoder(d2l.Encoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super().__init__()
        # embedding layer, think it as somewhat similar to one-hot previously told.
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)
    def forward(self, X, *args):
        """shape of output X (batch_size, num_steps, embed_size)"""
        X = self.embedding(X)
        """need to swtich dimentions to make num_steps the first dimention"""
        X = X.permute(1, 0, 2)
        """shape of output (num_steps,batch_size,num_hiddens), since it's encoder, no need to use a classifier"""
        """shape of state (num_layers,batch_size,num_hiddens)"""
        output, state = self.rnn(X)
        return output, state
    