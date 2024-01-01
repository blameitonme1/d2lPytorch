import math
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import languageModel
batch_size , num_steps = 32, 35
train_iter, vocab = languageModel.load_data_time_machine(batch_size, num_steps)
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

def predict_ch8(prefix, num_preds, net, vocab, device):
    """predict new characters after prefix"""
    """ batch_size is how many sentence given to predict , time step is 1"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda : torch.tensor([outputs[-1]], device=device).reshape((1,1))
    for y in prefix[1:]: # warm up phase
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        # using the predicted result to yet predict more words.
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    # return the predicted words.
    return ' '.join([vocab.idx_to_token[i] for i in outputs])

def grad_clipping(net, theta):
    """ control the gradient to prevent them to big or to small, the threshold is theta"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)
    for X, Y in train_iter:
        if state is None or use_random_iter:
            """init state when first iteration or use randomly iterator"""
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state is a tensor to nn.GRU
                state.detach_()
            else:
                # state is a tensor for LSTM or our scratch model
                for s in state:
                    s.detach_()
        # transpose y to make it's shape same as y-hat
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        # cross entropy loss
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
     # return perplexity
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    # calculating perplexity
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size=batch_size)
    predict = lambda prefix : predict_ch8(prefix, 50, net, vocab, device)
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter
        )
        if (epoch + 1) % 120 == 0:
            print((predict('time traveller')))
            animator.add(epoch + 1, [ppl])
    print(f'perplexity : {ppl:.1f}, {speed:.1f} tokens per second {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
num_epochs, lr = 500, 1
num_hiddens = 512
net = RNNModelFromScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                    init_rnn_state, rnn)
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())