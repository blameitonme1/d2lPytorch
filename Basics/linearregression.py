import torch
import numpy as np
from torch.utils import data
from d2l import torch as d2l
# nn是神经网络的缩写
from torch import nn

net = nn.Sequential(nn.Linear(2, 1))

#加载数据集
def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

next(iter(data_iter))
# 初始化权重和bias
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
# 损失算法也可以直接调库
loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')