import torch
a = torch.tensor([[1, 2], [4, 5], [7, 8]])
a.shape
a.sum().shape
a.sum(axis = [0])
a.sum(axis = [0,1]).shape