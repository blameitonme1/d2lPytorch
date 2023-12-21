import torch
import torchvision
from torch import nn
from d2l import torch as d2l
import csv
import pandas as pd
import numpy as np
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
# change the classifier of a pretrained model
class ModifiedVGG16(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model = model = torchvision.models.vgg16(pretrained=True)
        self.features = model.features
        self.classifer = torch.nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 176)
        )
    def forward(self, X):
        X = self.features(X)
        X = self.classifer(X)
        return X

data_dir = "D:/DeepLearning/data/classify-leaves/classify-leaves"

class LeaveDataset(Dataset):
    def __init__(self, dataframe, transform=None) -> None:
        super().__init__()
        self.imgs = dataframe['image']
        self.labels = dataframe['label']
        self.transform = transform
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        relative_path = self.imgs.iloc[idx]
        img = Image.open(os.path.join(data_dir, relative_path))
        id = relative_path.split('/')[-1].split('.')[0]
        label = self.labels.iloc[idx]
        if self.transform:
            img = self.transform(img)
        return img, cat_to_index[label]

class predDataset(Dataset):
    def __init__(self, dataframe, transform=None) -> None:
        super().__init__()
        self.imgs = dataframe['image']
        self.transform = transform
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        relative_path = self.imgs.iloc[idx]
        img = Image.open(os.path.join(data_dir, relative_path))
        if self.transform:
            img = self.transform(img)
        return img, relative_path
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
])
test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.ToTensor()
])
net = ModifiedVGG16().cuda()
df1 = pd.read_csv("D:\\DeepLearning\\data\\classify-leaves\\classify-leaves\\train.csv")
categories = df1['label'].unique().tolist()
cat_to_index = {leaf_name : index for index, leaf_name in enumerate(categories)}
# random_state is a seed for random number generator
train_df, val_df = train_test_split(df1, test_size=0.2, random_state=42)
train_dataset = LeaveDataset(train_df, transform=train_augs)
test_dataset = LeaveDataset(val_df, transform=test_augs)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
loss = nn.CrossEntropyLoss()
trainer = torch.optim.Adam(net.parameters(), lr=0.001)
# finetuning a liitle bit
d2l.train_ch13(net, train_iter=train_loader, test_iter=test_loader, loss=loss, trainer=trainer, num_epochs=5)

df2 = pd.read_csv("D:\DeepLearning\data\classify-leaves\classify-leaves\test.csv")
pred_dataset = predDataset(df2, transform=test_augs)
pred_loader = DataLoader(pred_dataset, batch_size=64, shuffle=False)
pred = []
net.eval()
with torch.no_grad():
    for i, (batch, names) in enumerate(pred_loader):
        batch = batch.cuda()
        batch = net(batch)
        _, pred_labels = torch.max(batch, 1)
        labels = [cat_to_index[index.item()] for index in pred_labels]
        pred.extend(list(zip(names, labels)))

with open('submissions.csv', 'w', newline='') as csvfile:
    fieldnames = ['image', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for i, (image_name, label) in enumerate(pred):
        #writer.writerow({'id': i + 1, 'label': label})
        writer.writerow({'image': image_name, 'label': label})


