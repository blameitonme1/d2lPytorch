import torch
from d2l import torch as d2l
import pandas as pd
from torch import nn
import numpy as np
test = pd.read_csv("D:\DeepLearning\data\CAhousePredictiontest.csv")
train = pd.read_csv("D:\DeepLearning\data\CAhousePredictiontrain.csv")
train_data = train.drop(['Id', 'Sold Price', 'State', 'Summary', 'Address'],axis=1)
test_data = test.drop(['Id', 'State', 'Summary', 'Address'], axis=1)
print(train_data.shape)
print(test_data.shape)
# 注意使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
all_features = pd.concat((train_data.iloc[:, :], test_data.iloc[:, :]))
print(all_features.shape)
# 数据预处理, 将数据标准化
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 将所有缺失值赋值0
all_features[numeric_features] = all_features[numeric_features].fillna(0)
# 参考了别人的代码里面，将这两列转化为pandas的标准dateTime类型，方便进行数据分析
all_features['Listed On'] = pd.to_datetime(all_features['Listed On'], format="%Y-%m-%d")
all_features['Last Sold On'] = pd.to_datetime(all_features['Last Sold On'], format="%Y-%m-%d")
# upHalf_features = all_features.iloc[0:all_features.shape[0] // 2, :]
# lowHalf_features = all_features.iloc[all_features.shape[0] // 2 : , :]
features = list(numeric_features)
features.append('Type') # Type 的种类不是很多，可以加上去 
print(features)
# 基本上只选取了类型是数字的类型，不然模型参数太多，没有足够的内存
all_features = all_features[features]
all_features = pd.get_dummies(all_features, dummy_na=True) # 独热编码
all_features = all_features * 1 # 确保bool转化为数值
print(all_features.shape)
n_train = train_data.shape[0]
# 训练数据和测试数据转化为张量，准备开始训练
train_features = torch.tensor(all_features[: n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train : ].values, dtype=torch.float32)
train_labels = torch.tensor(
    train['Sold Price'].values.reshape(-1, 1), dtype=torch.float32)
# 终于可以开始训练
loss = nn.MSELoss()
in_features = train_features.shape[1]
print(in_features)
# 定义一个基线模型MLP
def get_net():
    return nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
    )
# 评价正确率的标准，这样不会被数据分布影响过大，因为取了对数
def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf')) # 模型的预测值
    return torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels))).item() # 这里读取的就是一个标量，所以加上item()
# 训练方法，直接使用d2l的
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            X=X.to(device)
            y=y.to(device)
            optimizer.zero_grad()
            l = loss(net(X), y).to(device)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid
# 使用k折交叉验证，因为是一个小数据集，没有足够的验证数据集
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net().to(device)
        # 注意这里*data是将data解包成原来的样子
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k
train_features=train_features.to(device)
train_labels=train_labels.to(device)
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 0.01, 0, 512
# train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
#                           weight_decay, batch_size)
# print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
#       f'平均验证log rmse: {float(valid_l):f}')
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net().to(device)
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    # 将网络应用于测试集。
    preds = net(test_features).detach().cpu().numpy()
    # 将其重新格式化以导出到Kaggle
    test_data['Sold Price'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test['Id'], test_data['Sold Price']], axis=1)
    submission.to_csv('CAhousePredictionSubmission.csv', index=False)
test_features = test_features.to(device)
train_features = train_features.to(device)
train_labels = train_labels.to(device)
train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)
