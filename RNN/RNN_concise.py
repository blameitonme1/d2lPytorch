import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import languageModel
import textPreprocessing
batch_size, num_steps = 32, 35
train_iter, vocab = languageModel.load_data_time_machine()
num_hiddens = 256
rnn_layers = nn.RNN(len(vocab), num_hiddens)
state = torch.zeros((1, batch_size, num_hiddens))
"""already tranposed X"""
X = torch.rand(size=(num_steps, batch_size, len(vocab)))
class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size) -> None:
        super().__init__()
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(input.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        """manually change the dimensions of Y to (num_steps * batch_size, len(vocab))"""
        output = self.linear(Y.reshape(-1, Y.shape[-1]))
        return output, state
    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            return torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens),
                device=device
            )
        else:
            # using a tuple when LSTM
            return (torch.zeros((
            self.num_directions * self.rnn.num_layers,
            batch_size, self.num_hiddens), device=device),
                torch.zeros((
                    self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens), device=device))
