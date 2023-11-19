import torch
from torch import nn
from d2l import torch as d2l
from softmaxScratch import train_ch3

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

# loss函数
net.apply(init_weight)
loss = nn.CrossEntropyLoss()
# 优化器
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
num_epochs = 10
# 开始训练
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
d2l.plt.show()