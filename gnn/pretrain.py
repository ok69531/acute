#%%
import os
import sys
sys.path.append('../')

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn, optim

from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.loader import DataLoader

from module.model import GNNGraphPred

import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300


#%%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset_name = 'lipo'
dataset = PygGraphPropPredDataset(f'ogbg-mol{dataset_name}')
split_idx = dataset.get_idx_split()

train_loader = DataLoader(dataset[split_idx['train']], batch_size = 32, shuffle = True, num_workers = 0)
val_loader = DataLoader(dataset[split_idx['valid']], batch_size = 32, shuffle = False, num_workers = 0)
test_loader = DataLoader(dataset[split_idx['test']], batch_size = 32, shuffle = False, num_workers = 0)


#%%
def train(model, device, loader, criterion, optimizer):
    model.train()

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        running_loss = 0.0
        
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            
            optimizer.zero_grad()
            is_labeled = batch.y == batch.y
            loss = criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss.backward(create_graph = True)
            
            optimizer.step()


@torch.no_grad()
def evaluation(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


#%%
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

num_task = 1
num_layer = 5
emb_dim = 300
gnn_type = 'gin'
graph_pooling = 'mean'
dropout_ratio = 0.3
JK = 'last'
virtual = False
residual = False 
lr = 0.001

model = GNNGraphPred(num_tasks = num_task, num_layer = num_layer, 
                     emb_dim = emb_dim, gnn_type = gnn_type,
                     graph_pooling = graph_pooling, drop_ratio = dropout_ratio, JK = JK, 
                     virtual_node = virtual, residual = residual)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr = lr)
evaluator = Evaluator(f'ogbg-mol{dataset_name}')
criterion = nn.MSELoss()

best_epoch = 0
best_val_loss = 1e+7
best_test_loss = 0
for epoch in range(1, 300+1):
    print(f'=== epoch {epoch}')
    
    train(model, device, train_loader, criterion, optimizer)
    # aa_train(model, device, train_loader, criterion, optimizer)
    
    train_loss = evaluation(model, device, train_loader, evaluator)[dataset.eval_metric]
    val_loss = evaluation(model, device, val_loader, evaluator)[dataset.eval_metric]
    test_loss = evaluation(model, device, test_loader, evaluator)[dataset.eval_metric]
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_test_loss = test_loss
        best_epoch = epoch
        
        model_params = model.state_dict()
        
    print(f'train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, test loss: {test_loss:.4f}')

path = f'{dataset_name}.pth'
gnn_check_points = {'epoch': best_epoch,
                    'gnn_model_state_dict': model_params,
                    'optimizer_state_dict': optimizer.state_dict()}
torch.save(gnn_check_points, path)

#%%
model.eval()

test_pred = []
for _, batch in enumerate(test_loader):
    batch = batch.to(device)
    
    pred = model(batch)
    test_pred.append(pred)

test_pred = torch.cat(test_pred).view(-1).detach()
test_y = dataset[split_idx['test']].y.view(-1)


# %%
l = torch.linspace(min(torch.cat([test_pred, test_y])), max(torch.cat([test_pred, test_y])), 100)

plt.figure(figsize = (7, 7))
plt.plot(l, l)
plt.scatter(test_pred, test_y)
plt.xlabel('pred')
plt.ylabel('true')
plt.show()
plt.close()

# %%
