import pandas as pd
import numpy as np
from src.tab_data_oracle import *
from utils.pre_process_tab import load_preprocess
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.autograd.functional import jacobian
from sklearn.model_selection import train_test_split
from src.oracle import oracle_pred

TARGET= 'over_50k'

device = torch.device('cpu')

class DNN(nn.Module):
    def __init__(self, input_space):
        super().__init__()
        self.net= nn.Sequential(
            nn.Linear(input_space,150),
            nn.ReLU(),
            nn.Linear(150, 50),
            nn.ReLU(),
            nn.Linear(50,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.net(x)    

class PandasDataSet(Dataset):
    def __init__(self, attack_dataset, oracle_preds):
        super().__init__()
        self.pandas_dataset= torch.FloatTensor(attack_dataset.values)
        self.oracle_preds= oracle_preds
    
    def __len__(self):
        return len(self.oracle_preds)
    
    def __getitem__(self, idx):
        return self.pandas_dataset[idx], torch.Tensor([self.oracle_preds[idx]])

class LabeledAttackSet(Dataset):
    def __init__(self, attack_dataset, oracle_preds):
        super().__init__()
        self.attack_dataset= attack_dataset
        self.oracle_preds= oracle_preds
    
    def __len__(self):
        return len(self.oracle_preds)
    
    def __getitem__(self, idx):
        return self.attack_dataset[idx][0], torch.Tensor([self.oracle_preds[idx]])

class UnlabeledAttackSet(Dataset):
    def __init__(self, attack_dataset):
        super().__init__()
        self.attack_dataset= attack_dataset
    
    def __len__(self):
        return self.attack_dataset.shape[0]
    
    def __getitem__(self, idx):
        return self.attack_dataset[idx], torch.zeros(1)

def train_network(net, epochs,train_dataloader, valid_dataloader, optim, criterion, pth):
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
                predicted = np.where(pred.detach().numpy()>.5, 1,0)
                correct+= (predicted==target.numpy()).sum().item()
                total += target.size(0)
        print('epoch {}/{} training loss: {}, valid loss: {}, valid accuracy: {}'.format(epoch, epochs, total_loss/ nb_batch_train, total_loss_valid/ nb_batch_valid, correct/total ))
        if total_loss_valid/ nb_batch_valid < best_loss:
            print('best loss improove from {} to {} saving model'.format(best_loss,total_loss_valid/ nb_batch_valid ))
            best_loss= total_loss_valid/ nb_batch_valid
            torch.save(net, pth) #put as param

def create_synth_example(substitute, dataset, lmbd=.1):
    synth=[]
    for i in dataset:
        x,y = i
        j= jacobian(substitute, x)
        x_synth= x + lmbd* torch.sign(j[0,:].view((107)))
        synth.append(x_synth.view(1,107))
    return torch.cat(synth)

def augment_dataset(attack_dataset, substitute, oracle, lmbd=.1):
    x_synth= create_synth_example(substitute, attack_dataset, lmbd)
    unlabeled_attack_set= UnlabeledAttackSet(x_synth)
    #unlabeled_attack_dataloader= DataLoader(unlabeled_attack_set, batch_size=64, shuffle=False)
    oracle_preds= oracle.oracle_pred(x_synth.cpu().detach().numpy(), np.zeros(x_synth.shape[0]))
    labeled_synth_set= LabeledAttackSet(unlabeled_attack_set, oracle_preds)
    return torch.utils.data.ConcatDataset([attack_dataset, labeled_synth_set])


def substitute_training(attack_set, test_dataloader, oracle, lmbd=.1, lr=0.1, substitute_learning_iter=6, pth='./models/substitute_tab_gb.pth'):
    criterion = nn.BCELoss()
    for _ in range(substitute_learning_iter):
        attack_dataloader= DataLoader(attack_set, batch_size=64, shuffle=True)
        substitute= DNN(107)
        optim= torch.optim.SGD(substitute.parameters(), lr= lr, momentum=0.9)
        train_network(substitute, 10, attack_dataloader, test_dataloader, optim, criterion, pth)
        attack_set= augment_dataset(attack_set, substitute, oracle, lmbd)
    return substitute


if __name__=='__main__':
    train_x, test_x, train_y, test_y= load_preprocess("/home/hadrien/data/adult/adult.csv", ['fnlwgt'], TARGET)
    train_x, valid_x, train_y, valid_y= train_test_split(train_x, train_y, train_size=.8, random_state=42)
    attack_set_o_x, test_set_x, attack_set_o_y, test_set_y= train_test_split(test_x, test_y, train_size=150, random_state=42)

    with open("./models/gb", "rb") as fp:   # Unpickling
        oracle = pickle.load(fp)

    oracle.oracle_pred(valid_x, valid_y, True)

    #attack_dataloader_o= DataLoader(attack_set_o,batch_size=64, shuffle=False)
    oracle_preds= oracle.oracle_pred(attack_set_o_x, attack_set_o_y)

    attack_set= PandasDataSet(attack_set_o_x, oracle_preds)

    test_set= PandasDataSet(test_set_x, test_set_y.to_list())
    test_dataloader= DataLoader(test_set,64,False)

    substitute= substitute_training(attack_set, test_dataloader, oracle)

    print('oracle perf:')
    oracle.oracle_pred(test_set_x, test_set_y, True)
    print('substitute perf:')
    with torch.no_grad():
        correct=0
        total=0
        for batch in test_dataloader:
            feature, target= batch[0], batch[1]
            pred= substitute(feature)
            predicted = np.where(pred.detach().numpy()>.5, 1,0)
            correct+= (predicted==target.numpy()).sum().item()
            total += target.size(0)
    print('accuracy:', correct/total)