import math
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import languageModel
import RNN_scratch
import RNN_concise
batch_size , num_steps = 32, 35
train_iter, vocab = languageModel.load_data_time_machine(batch_size, num_steps)
# using PyToch APIs
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs =vocab_size
device = d2l.try_gpu()
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
model = RNN_concise.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
num_epochs, lr = 500, 2
RNN_scratch.train_ch8(model, train_iter, vocab, lr*1.0, num_epochs, device)