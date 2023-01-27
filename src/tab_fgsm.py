import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
from torchvision import datasets, transforms
import torch.nn.functional as F
#from torch.autograd.functional import jacobian
import pandas as pd
import numpy as np
from src.tab_data_oracle import *
from utils.pre_process_tab import load_preprocess
from sklearn.model_selection import train_test_split
from src.tab_data_substitute import PandasDataSet, DNN
from src.tab_data_oracle import *


device = torch.device('cpu')

EPS= .3

def create_adv(dataloader, substitute, criterion, eps= EPS):
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

TARGET= 'over_50k'
train_x, test_x, train_y, test_y= load_preprocess("/home/hadrien/data/adult/adult.csv", ['fnlwgt'], TARGET)
train_x, valid_x, train_y, valid_y= train_test_split(train_x, train_y, train_size=.8, random_state=42)
attack_set_o_x, test_set_x, attack_set_o_y, test_set_y= train_test_split(test_x, test_y, train_size=150, random_state=42)

with open("./models/gb", "rb") as fp:   # Unpickling
    oracle = pickle.load(fp)
#oracle.oracle_pred(valid_x, valid_y, True)
substitute= torch.load('./models/substitute_tab_gb.pth')


test_set= PandasDataSet(test_set_x, test_set_y.to_list())
test_dataloader= DataLoader(test_set,64,False)
criterion = nn.BCELoss()
adv_x= create_adv(test_dataloader, substitute, criterion)
#adv_dataset= AdvDataset(adv_x, test_set)
#adv_dataloader= DataLoader(adv_dataset, batch_size=64, shuffle=False)
#oracle_pred(oracle, adv_dataloader, True)

print('oracle perf on clean data:')
oracle.oracle_pred(test_set_x, test_set_y, True)
print('oracle perf on adverserial example:')
oracle.oracle_pred(adv_x.detach().numpy(), test_set_y, True)

#lr:
#oracle perf on clean data:
#lr oracle accuracy: 0.8557022559517622
#oracle perf on adverserial example:
#lr oracle accuracy: 0.29348165089926187
#
#tree:
#oracle perf on clean data:
#tree oracle accuracy: 0.7554839380392973
#oracle perf on adverserial example:
#tree oracle accuracy: 0.7453997297016322
#
#rf:
#oracle perf on clean data:
#rf oracle accuracy: 0.725023391204907
#oracle perf on adverserial example:
#rf oracle accuracy: 0.7199293065807256
#
#svm
#oracle perf on clean data:
#lr oracle accuracy: 0.8580933568978064
#oracle perf on adverserial example:
#lr oracle accuracy: 0.7225283293481651
#
#Light Gradient boosting
#oracle perf on clean data:
#gb oracle accuracy: 0.7554839380392973
#oracle perf on adverserial example:
#gb oracle accuracy: 0.33090757875038984
