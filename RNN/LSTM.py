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
"""remember that gate'a value is between 0 and 1"""
def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size
    def normal(shape):
        return torch.rand(size=shape, device=device) * 0.01
    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))
    W_xi, W_hi, b_i = three() # input gate
    W_xf, W_hf, b_f = three() # forget gate
    W_xo, W_ho, b_o = three() # output gate
    W_xc, W_hc, b_c = three() # candidate memory cell
    # linear layer
    # linear layer
    W_hq = normal((num_hiddens, vocab_size))
    b_q = torch.zeros(num_outputs, device=device)
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params
def init_lstm_state(batch_size, num_hiddens, device):
    """a same shape memory cell added in the tuple"""
    return (torch.zeros((batch_size,num_hiddens),device=device),
            torch.zeros((batch_size,num_hiddens),device=device)
            )
def lstm(inputs, state, params):
    """forward function when using LSTM"""
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
     W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)
        Y = (H @ W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,C)
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = RNN_scratch.RNNModelFromScratch(len(vocab), num_hiddens, device, get_lstm_params,
                            init_lstm_state, lstm)
RNN_scratch.train_ch8(model, train_iter, vocab, lr, num_epochs, device)