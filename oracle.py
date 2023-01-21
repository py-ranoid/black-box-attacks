import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
from torchvision import datasets, transforms
import torch.nn.functional as F

if torch.backends.mps.is_available(): # Mac M1/M2
    device = torch.device('mps')
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# contains 60k images from the MNIST train dataset by pytorch
train_dataset= datasets.MNIST(
         "/home/hadrien/data/mnist",
         train=True,
         download=True,
         transform=transforms.Compose(
             [ transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
         ),
     )

valid_dataset= datasets.MNIST(
         "/home/hadrien/data/mnist",
         train=False,
         download=True,
         transform=transforms.Compose(
             [ transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
         ),
     )

train_dataloader= DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
valid_dataloader= DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=8)




class CNN(nn.Module):
    def __init__(self, model_pth):
        super().__init__()
        self.model_pth= model_pth
        self.conv_block= nn.Sequential(
            nn.Conv2d(1,32,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,2),
            nn.MaxPool2d(2),
        )
        self.linear_1= nn.Linear(64*6*6, 200)
        self.linear_2= nn.Linear(200,200)
        self.logit= nn.Linear(200,10)
    
    def forward(self,x):
        x= self.conv_block(x)
        x= x.view(-1, 64*6*6)
        x= F.relu(self.linear_1(x))
        x= F.relu(self.linear_2(x))
        return F.softmax(self.logit(x))
    



def train_network(net, epochs,train_dataloader, valid_dataloader, optim, criterion):
    nb_batch_train= len(train_dataloader)
    nb_batch_valid= len(valid_dataloader)
    best_loss= 1000000
    for epoch in range(epochs):
        total_loss=0
        total_loss_valid=0
        correct=0
        total=0
        for i, batch in enumerate(tqdm(train_dataloader)):
            feature, target= batch[0], batch[1]
            feature.to(device)
            target.to(device)
            optim.zero_grad()
            pred= net(feature)
            loss= criterion(pred, target)
            loss.backward()
            optim.step()
            total_loss += loss.item()
        with torch.no_grad():
            for batch in valid_dataloader:
                feature, target= batch[0], batch[1]
                feature.to(device)
                target.to(device)
                pred= net(feature)
                loss= criterion(pred, target)
                total_loss_valid+= loss
                _, predicted = torch.max(pred.data, 1)
                correct+= (predicted==target).sum().item()
                total += target.size(0)
        print('epoch {}/{} training loss: {}, valid loss: {}, valid accuracy: {}'.format(epoch, epochs, total_loss/ nb_batch_train, total_loss_valid/ nb_batch_valid, correct/total ))
        if total_loss_valid/ nb_batch_valid < best_loss:
            print('best loss improove from {} to {} saving model'.format(best_loss,total_loss_valid/ nb_batch_valid ))
            best_loss= total_loss_valid/ nb_batch_valid
            torch.save(net, './oracle.pth') #put as param

oracle= CNN('./oracle.pth')
optim= torch.optim.Adam(oracle.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
train_network(oracle, 10, train_dataloader, valid_dataloader, optim, criterion)
