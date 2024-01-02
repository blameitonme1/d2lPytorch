import math
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import languageModel
import RNN_scratch
batch_size , num_steps = 32, 35
train_iter, vocab = languageModel.load_data_time_machine(batch_size, num_steps)
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size
    def normal(shape):
        return torch.rand(size=shape, device=device) * 0.01
    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))
    W_xz, W_hz, b_z = three() # update gate
    W_xr, W_hr, b_r = three() # reset gate
    W_xh, W_hh, b_h = three() # candidate hidden state
    # linear layer
    W_hq = normal((num_hiddens, vocab_size))
    b_q = torch.zeros(num_outputs, device=device)
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params
def init_gru_state(batch_size, num_hiddens, device):
    # returns a tuple considering LSTM
    return (torch.zeros((batch_size, num_hiddens), device=device),)

"""gru's forward function"""
def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H,  = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_candidate = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_candidate
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)

# training a little bit
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = RNN_scratch.RNNModelFromScratch(len(vocab), num_hiddens, device, get_params,
                            init_gru_state, gru)
RNN_scratch.train_ch8(model, train_iter, vocab, lr, num_epochs, device)