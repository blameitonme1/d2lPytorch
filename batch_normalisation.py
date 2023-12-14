import torch
from torch import nn
from d2l import torch as d2l
# 参照笔记 DeepLearning convolution batch normalization
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 判断训练模式还是预测模式,eps防止除0
    if not torch.is_grad_enabled():
        # 预测模式，直接使用传入的移动平均所得的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2,4)
        if len(X.shape) == 2:
            # 全连接层输出，批量维度求均值和方差
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 卷积层输出， keepdim因为利用广播机制
            mean = X.mean(dim=(0,2,3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0,2,3), keepdim=True)
        # 训练模式，用当前的均值和方差
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新 moving mean和 moving var
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta
    # 注意 tensor.data 返回不追踪梯度的 tensor
    return Y, moving_mean.data, moving_var.data

class BatchNorm(nn.Module):
    # num_features 输出数量或者卷积层的输出通道数
    # num_dim 2 表示全连接层， 4表示卷积层
    def __init__(self, num_features, num_dims) -> None:
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 初始化scale和shift
        self.gamma = torch.Parameter(torch.ones(shape))
        self.beta = torch.Parameter(torch.zeros(shape))
        # 非模型参数
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)
    
    def forward(self, X):
        # X不在内存就将mean和var移动到显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新之后的moving_mean和moving_var
        Y , self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean, self.moving_var, eps=1e-5, momentum=0.9)
    
        return Y