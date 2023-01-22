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
from torch.autograd.functional import jacobian

if torch.backends.mps.is_available(): # Mac M1/M2
    device = torch.device('mps')
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

test_dataset= datasets.MNIST(
         "/home/hadrien/data/mnist",
         train=False,
         download=True,
         transform=transforms.Compose(
             [ transforms.ToTensor()]#, transforms.Normalize([0.5], [0.5])]
         ),
     )

class LabeledAttackSet(Dataset):
    def __init__(self, attack_dataset, oracle_preds):
        super().__init__()
        self.attack_dataset= attack_dataset
        self.oracle_preds= oracle_preds
    
    def __len__(self):
        return len(self.oracle_preds)
    
    def __getitem__(self, idx):
        return self.attack_dataset[idx][0], self.oracle_preds[idx]

class UnlabeledAttackSet(Dataset):
    def __init__(self, attack_dataset):
        super().__init__()
        self.attack_dataset= attack_dataset
    
    def __len__(self):
        return self.attack_dataset.shape[0]
    
    def __getitem__(self, idx):
        return self.attack_dataset[idx], torch.zeros(1)

def create_synth_example(substitute, dataset, lmbd=.1):
    synth=[]
    for i in dataset:
        x,y = i
        j= jacobian(substitute, x)
        x_synth= x + lmbd* torch.sign(j[:,y,:,:].view((1,1,28,28)))
        synth.append(x_synth)
    return torch.cat(synth)

def augment_dataset(attack_dataset, substitute, oracle, lmbd=.1):
    x_synth= create_synth_example(substitute, attack_dataset, lmbd)
    unlabeled_attack_set= UnlabeledAttackSet(x_synth)
    unlabeled_attack_dataloader= DataLoader(unlabeled_attack_set, batch_size=64, shuffle=False)
    oracle_preds= oracle_pred(oracle, unlabeled_attack_dataloader)
    labeled_synth_set= LabeledAttackSet(unlabeled_attack_set, oracle_preds)
    return torch.utils.data.ConcatDataset([attack_dataset, labeled_synth_set])


def substitute_training(attack_set, test_dataloader, oracle, lmbd=.1, lr=0.01, substitute_learning_iter=6, pth='./substitute.pth'):
    criterion = nn.CrossEntropyLoss()
    for _ in range(substitute_learning_iter):
        attack_dataloader= DataLoader(attack_set, batch_size=64, shuffle=True)
        substitute= CNN()
        optim= torch.optim.SGD(substitute.parameters(), lr= lr, momentum=0.9)
        train_network(substitute, 10, attack_dataloader, test_dataloader, optim, criterion, pth)
        attack_set= augment_dataset(attack_set, substitute, oracle, lmbd)
    return substitute

attack_set_o, test_set= torch.utils.data.random_split(test_dataset, [150,9850],generator=torch.Generator().manual_seed(42))

oracle= CNN()
oracle= torch.load('./oracle.pth')

attack_dataloader_o= DataLoader(attack_set_o,batch_size=64, shuffle=False)
oracle_preds=oracle_pred(oracle, attack_dataloader_o)

attack_set= LabeledAttackSet(attack_set_o, oracle_preds)
test_dataloader= DataLoader(test_set,64,False)

substitute= substitute_training(attack_set, test_dataloader, oracle)

print('oracle perf:')
oracle_pred(oracle, test_dataloader, True)
print('substitute perf:')
oracle_pred(substitute, test_dataloader, True)