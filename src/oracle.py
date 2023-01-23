import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision import datasets, transforms
import torch.nn.functional as F

if torch.backends.mps.is_available(): # Mac M1/M2
    device = torch.device('mps')
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
ORACLE_MODEL_PATH = './models/oracle.pth'

class CNN_min(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block= nn.Sequential(
            nn.Conv2d(1,16,2),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,2),
            nn.MaxPool2d(2),
        )
        self.linear_1= nn.Linear(32*6*6, 200)
        self.dropout = nn.Dropout(0.2)        
        self.linear_2= nn.Linear(200,100)
        self.logit= nn.Linear(100,10)
    
    def forward(self,x):
        x= self.conv_block(x)
        x= x.view(-1, 32*6*6)
        x= F.relu(self.linear_1(x))
        x= self.dropout(x)
        x= F.relu(self.linear_2(x))
        return F.softmax(self.logit(x))
    
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
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
    
def train_network(net, epochs, train_dataloader, valid_dataloader, optim, criterion, pth):
    nb_batch_train= len(train_dataloader)
    nb_batch_valid= len(valid_dataloader)
    best_loss= 1000000
    train_history = []
    for epoch in range(epochs):
        total_loss=0
        total_loss_valid=0
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
        
        _, val_acc, val_loss = oracle_pred_eval(net, criterion, valid_dataloader, evaluate=True)
        _, train_acc, train_loss = oracle_pred_eval(net, criterion, train_dataloader, evaluate=True)
                
        epoch_stats = {"epoch":epoch, "train_loss": train_loss, "val_loss":val_loss, "val_acc":val_acc, "train_acc":train_acc}
        train_history.append(epoch_stats)
        
        print('epoch {}/{} training loss: {}, valid loss: {}, valid accuracy: {}'.format(epoch_stats['epoch'], epochs, epoch_stats['train_loss'], epoch_stats['val_loss'], epoch_stats['val_acc'] ))
        if total_loss_valid/ nb_batch_valid < best_loss:
            print('best loss improove from {} to {} saving model'.format(best_loss,total_loss_valid/ nb_batch_valid ))
            best_loss= total_loss_valid/ nb_batch_valid
            torch.save(net, pth) #put as param
    return train_history

def oracle_pred_eval(net, criterion, dataloader, evaluate=False):
    preds=[]
    correct, total, loss = 0.0,0.0,0.0
    with torch.no_grad():
        for batch in dataloader:
            #Generate predictions for given batch in dataloader
            feature, target= batch[0], batch[1]
            feature.to(device)
            target.to(device)
            pred = net(feature)
            if evaluate:
                loss += criterion(pred, target)
            
            #Pick predicted class by max prob
            _, predicted = torch.max(pred.data, 1)
            preds.append(predicted)
            
            #Evaluate against target
            if evaluate:
                correct+= (predicted==target).sum().item()
                total += target.size(0)
    if evaluate:
        return torch.cat(preds), correct/total, float(loss/total)
    else:
        return torch.cat(preds)

if __name__=='__main__':
    # contains 60k images from the MNIST train dataset by pytorch
    train_dataset= datasets.MNIST(
            "./data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [ transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        )

    test_dataset= datasets.MNIST(
            "./data/mnist",
            train=False,
            download=True,
            transform=transforms.Compose(
                [ transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        )

    #Loading and spltting MNIST dataset
    train_dataset, valid_dataset= torch.utils.data.random_split(train_dataset, [50000,10000])
    train_dataloader= DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
    valid_dataloader= DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=8)

    #Initialising Oracle CNN, optimiser and criterion
    oracle = CNN()
    optim = torch.optim.Adam(oracle.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    #Load Oracle model if exists else train
    if os.path.exists(ORACLE_MODEL_PATH):
        oracle = torch.load(ORACLE_MODEL_PATH)
    else:
        train_history = train_network(oracle, 10, train_dataloader, valid_dataloader, optim, criterion, ORACLE_MODEL_PATH)
        print (pd.DataFrame(train_history))

    #Run predictions and evaluate
    preds, acc, loss = oracle_pred_eval(oracle, criterion, valid_dataloader, evaluate=True)
    print("validation accuracy : ",acc)
