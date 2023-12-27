import torch
from d2l import torch as d2l
from torch import nn

# use markov's model to predict future output
tau = 4
T = 1000
# features = [【x1, x2, x3, x4】,
#             【x2, x3, x4, x5】,
#                    ...
#            【x996, x997, x998, x999】
#             ]
# labels = [【x5】,
#            【x6】,
#             ...
#          【x1000】]
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
features = torch.zeros((T - tau, tau))
for i in range(tau):
    features[:, i] = x[i: T - tau + i]
labels = x[tau:].reshape((-1, 1))