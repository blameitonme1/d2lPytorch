import math
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
batch_size , num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
# using one-hot encoding
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01
    
    """ parameters in hidden layers"""
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens)) # without this line, it's just a normal MLP!
    b_h = torch.zeros(num_hiddens, device=device)
    # output layer
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # requires grad
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params
"""manually define the first state of the hidden layer, since no input before it."""
def init_rnn_state(batch_size, num_hiddens, device):
    # returns a tuple
    return (torch.zeros((batch_size, num_hiddens), device=device), )
def rnn(inputs, state, params):
    """shape of inputs (num_steps, batch_size, len(vocab))"""
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    """ shape of X (batch_size, len(vocab))"""
    for X in inputs:
        # update Hidden state
        H = torch.tanh(
            torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h
        )
        # generate output of this time step
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    # return current state for further utilizations.
    return torch.cat(outputs, dim=0), (H, )

""" define a class to pacakge these functions"""
class RNNModelFromScratch:
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn) -> None:
        # NOTICE THAT init_state is a function to init state, NOT an already inited state!
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_sate, self.forward_fn = init_state, forward_fn
    
    def __call__(self, X, state):
        # X is not yet been transposed.
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)
    def begin_state(self, batch_size, device):
        return self.init_sate(batch_size, self.num_hiddens, device)
        
        
