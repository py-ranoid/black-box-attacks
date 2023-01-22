import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
from torchvision import datasets, transforms
import torch.nn.functional as F
from oracle import CNN, oracle_pred, train_network
#from torch.autograd.functional import jacobian

if torch.backends.mps.is_available(): # Mac M1/M2
    device = torch.device('mps')
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

EPS= .3
EPS_SCALED= EPS*.5 + .5

def create_adv(dataloader, substitute, criterion, eps= EPS_SCALED):
    adv=[]
    for batch in dataloader:
        x, y= batch
        x.to(device)
        y.to(device)
        x.requires_grad=True
        pred= substitute(x)
        loss= criterion(pred, y)
        loss.backward()
        x_adv= x+ eps * torch.sign(x.grad)
        adv.append(x_adv)
    return torch.cat(adv)

class AdvDataset(Dataset):
    def __init__(self, adv_x, clean_dataset):
        super().__init__()
        self.adv_x= adv_x
        self.clean_dataset= clean_dataset
    
    def __len__(self):
        return self.adv_x.shape[0]
    
    def __getitem__(self, idx):
        return self.adv_x[idx], self.clean_dataset[idx][1]


substitute= torch.load('./substitute.pth')
oracle= torch.load('./oracle.pth')


test_dataset= datasets.MNIST(
         "/home/hadrien/data/mnist",
         train=False,
         download=True,
         transform=transforms.Compose(
             [ transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
         ),
     )

attack_set_o, test_set= torch.utils.data.random_split(test_dataset, [150,9850],generator=torch.Generator().manual_seed(42))
test_dataloader= DataLoader(test_set,64,False)
criterion = nn.CrossEntropyLoss()
adv_x= create_adv(test_dataloader, substitute, criterion)
adv_dataset= AdvDataset(adv_x, test_set)
adv_dataloader= DataLoader(adv_dataset, batch_size=64, shuffle=False)
oracle_pred(oracle, adv_dataloader, True)