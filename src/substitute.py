import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
from torchvision import datasets, transforms
import torch.nn.functional as F
from oracle import CNN, oracle_pred_eval, train_network, ORACLE_MODEL_PATH, CNN_min
from torch.autograd.functional import jacobian
import matplotlib.pyplot as plt

if torch.backends.mps.is_available(): # Mac M1/M2
    device = torch.device('mps')
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
SEED=121
SUBSTITUTE_MODEL_PATH='./models/substitute_min.pth'
torch.manual_seed(SEED)

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
    oracle_preds = oracle_pred_eval(oracle, None, unlabeled_attack_dataloader, evaluate=False)
    labeled_synth_set= LabeledAttackSet(unlabeled_attack_set, oracle_preds)
    return torch.utils.data.ConcatDataset([attack_dataset, labeled_synth_set])


def substitute_training(attack_set, test_dataloader, oracle, lmbd=.1, lr=0.01, substitute_learning_iter=6, pth=SUBSTITUTE_MODEL_PATH):
    criterion = nn.CrossEntropyLoss()
    all_training_hist = pd.DataFrame()
    for run_id in range(substitute_learning_iter):
        attack_dataloader= DataLoader(attack_set, batch_size=64, shuffle=True)
        substitute= CNN_min()
        optim= torch.optim.SGD(substitute.parameters(), lr= lr, momentum=0.9)
        train_hist = train_network(substitute, 12, attack_dataloader, test_dataloader, optim, criterion, pth)
        train_hist = pd.DataFrame(train_hist)
        train_hist['run'] = run_id
        print (train_hist)
        all_training_hist = all_training_hist.append(train_hist)
        all_training_hist.to_csv('./data/sub_train_hist_%r_S%r_min.csv'%(run_id, SEED))
        attack_set= augment_dataset(attack_set, substitute, oracle, lmbd)
    return substitute, all_training_hist

if __name__ == "__main__":
    
    test_dataset= datasets.MNIST(
         "./data/mnist",
         train=False,
         download=True,
         transform=transforms.Compose(
             [ transforms.ToTensor()]#, transforms.Normalize([0.5], [0.5])]
         ),
     )

    attack_set_o, test_set= torch.utils.data.random_split(test_dataset, [250,9750],generator=torch.Generator().manual_seed(42))
    test_dataloader= DataLoader(test_set, 64, False)

    #Loading Oracle weights
    oracle= CNN()
    oracle= torch.load(ORACLE_MODEL_PATH)
    criterion = nn.CrossEntropyLoss()

    # Labelling attack dataset and training substitute model
    attack_dataloader_o= DataLoader(attack_set_o, batch_size=64, shuffle=False)
    oracle_preds, oracle_acc, oracle_loss = oracle_pred_eval(oracle, criterion, attack_dataloader_o, evaluate=True)
    attack_set= LabeledAttackSet(attack_set_o, oracle_preds)
    substitute, sub_history= substitute_training(attack_set, test_dataloader, oracle, substitute_learning_iter=10)
    
    #Plot substitute training history
    sub_history = sub_history.reset_index().rename(columns={'index':'run_ind'})    
    ax = plt.figure(figsize=(16, 8), dpi=100)
    sub_history['train_acc'].plot(label='Training accuracy', color='orange')
    sub_history['val_acc'].plot(label='Validation accuracy', color='blue')
    plt.title('Substitute training')
    plt.xlabel('epochs')
    plt.vlines(sub_history.drop_duplicates('run',keep='first')['run_ind'].to_list(), ymin=0,ymax=1, colors='black', ls='--', lw=2, label='run')    
    plt.legend()
    plt.show()
    
    _, oracle_test_acc, oracle_test_loss = oracle_pred_eval(oracle, criterion, test_dataloader, evaluate=True)    
    print('oracle perf:',oracle_test_acc)
    
    _, sub_test_acc, sub_test_loss = oracle_pred_eval(substitute, criterion, test_dataloader, evaluate=True)
    print('substitute perf:',sub_test_acc)