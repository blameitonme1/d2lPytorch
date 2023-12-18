import torch
from d2l import torch as d2l
import os
import torchvision
from torch import nn
import torch.utils.data as data
# 网络上下载热狗分类数据集
#@save
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip',
                         'fba480ffa8aa7e0febbb511d181409f899b9baa5')

data_dir = d2l.download_extract('hotdog')
# 创建两个实例读取训练和测试数据集
train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, "train"))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, "test"))
# 标准化RGB通道
normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
)
# 训练时数据增强实例, 因为高宽比不一，所以要随机水平翻转
# 注意将操作传入一个列表
train_aug = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        normalize
    ]
)
# 测试集先resize再中心裁剪
test_aug = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize([256, 256]),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        normalize
    ]
)
# 调用提前训练好的模型,注意finetuning的时候学习率设置小一点即可，因为几次迭代就会出现巨大的accuracy提升
finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
# 调用xavier初始化全连接层权重
nn.init.xavier_uniform_(finetune_net.fc.weight)
# param_group是拿来和每pretrained的模型进行对比的，就不用了
def train_finetuning(net, learning_rate, batch_size=128, num_epoches=5,
                     param_group=True):
    train_iter = data.DataLoader(
        torchvision.datasets.ImageFolder(
            os.path.join(data_dir, "train"), transform=train_aug
        ), batch_size, shuffle=True
    )
    test_iter = data.DataLoader(
        torchvision.datasets.ImageFolder(
            os.path.join(data_dir, "test"),
            transform=test_aug
        ),
        batch_size=batch_size
    )
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                              weight_decay=0.001)
    
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epoches, 
                   devices)




