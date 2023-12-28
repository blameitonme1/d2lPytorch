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
# global variable
cat_to_index = {}
# change the classifier of a pretrained model
class ModifiedVGG16(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model = torchvision.models.vgg16(pretrained=True)
        self.features = model.features
        # freeze these parameters to avoid them being updated.
        for param in self.features.parameters():
            param.requires_grad = False
        self.classifer = torch.nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 176)
        )
    def forward(self, X):
        X = self.features(X)
        X = self.classifer(X)
        return X

data_dir = "D:\\DeepLearning\\data\\classify-leaves\\classify-leaves"

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
        label = self.labels.iloc[idx]
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(cat_to_index[label])

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
def train_model(model, train_loader, test_loader, loss_fn, trainer, num_epochs=5, device='cuda'):
    #assign it to model! Otherwise can't exploit GPU
    model = model.to(device)
    
    for epoch in range(num_epochs):
        print(f"epoch {epoch + 1}")
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            trainer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            trainer.step()

            total_loss += loss.item()

            _, predictions = torch.max(outputs.data, 1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_accuracy = correct_predictions / total_samples

        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Training Loss: {train_loss:.4f}, '
              f'Training Accuracy: {train_accuracy:.4f}')

        # Evaluate on the test set
        test_accuracy = test_model(model, test_loader)
        print(f'Test Accuracy: {test_accuracy:.4f}')

    print('Training finished.')

def test_model(net, test_loader):
    net = net.cuda()
    net.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch, labels in test_loader:
            batch = batch.cuda()
            labels = labels.cuda()
            # Forward pass
            outputs = net(batch)

            # Calculate accuracy
            _, predictions = torch.max(outputs, 1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = (total_correct / total_samples)
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy

net = ModifiedVGG16().cuda()
df1 = pd.read_csv("D:\\DeepLearning\\data\\classify-leaves\\classify-leaves\\train.csv")
categories = df1['label'].unique().tolist()
cat_to_index = {leaf_name : index for index, leaf_name in enumerate(categories)}
# random_state is a seed for random number generator
train_df, val_df = train_test_split(df1, test_size=0.2, random_state=42)
train_dataset = LeaveDataset(train_df, transform=train_augs)
test_dataset = LeaveDataset(val_df, transform=test_augs)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
loss = nn.CrossEntropyLoss()
trainer = torch.optim.Adam(net.parameters(), lr=0.0001)
# finetuning a liitle bit
train_model(net, train_loader=train_loader, test_loader=test_loader, loss_fn=loss, trainer=trainer, num_epochs=10)
# test accuracy
test_model(net, test_loader)
df2 = pd.read_csv("D:\\DeepLearning\\data\\classify-leaves\\classify-leaves\\test.csv")
pred_dataset = predDataset(df2, transform=test_augs)
pred_loader = DataLoader(pred_dataset, batch_size=64, shuffle=False)
pred = []
net.eval()
with torch.no_grad():
    for i, (batch, names) in enumerate(pred_loader):
        batch = batch.cuda()
        batch = net(batch)
        _, pred_labels = torch.max(batch, 1)
        labels = [categories[index.item()] for index in pred_labels]
        pred.extend(list(zip(names, labels)))

with open('submissions.csv', 'w', newline='') as csvfile:
    fieldnames = ['image', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for i, (image_name, label) in enumerate(pred):
        #writer.writerow({'id': i + 1, 'label': label})
        writer.writerow({'image': image_name, 'label': label})


