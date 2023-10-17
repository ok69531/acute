#%%
import os
import sys
sys.path.append('../')

import numpy as np
import pandas as pd
from collections import OrderedDict

import torch
from torch import nn, optim

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)
from sklearn.model_selection import train_test_split

from module.model import GNNGraphPred
from module.mol import smiles2graph, read_graph_pyg

import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300

torch.autograd.set_detect_anomaly(True)


#%%
class AcuteDataset(InMemoryDataset):
    def __init__(self, root, classification = False, log_transform = False, transform = None, pre_transform = None, pre_filter = None):
        self.root = root
        self.classification = classification
        self.log_transform = log_transform
        
        super(AcuteDataset, self).__init__(self.root, transform, pre_transform, pre_filter)
        
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return file_name_list
    
    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'
    
    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')
    
    def process(self):
        df = pd.read_excel(self.raw_paths[0])
        
        smiles_list = list(df.SMILES)
        
        graph_list = [smiles2graph(x) for x in smiles_list]
        data_list = read_graph_pyg(graph_list)
        
        if self.classification:
            graph_target = df['category'].to_numpy()
        else:
            graph_target = df['value'].to_numpy()
        
        if self.log_transform:
            graph_target = np.log(graph_target)
        
        for i, g in enumerate(data_list):
            g.id = torch.tensor([i])
            
            if self.classification:
                g.y = torch.from_numpy(graph_target)[i].view(1, -1).to(torch.int64)
            else:
                g.y = torch.from_numpy(graph_target)[i].view(1, -1).to(torch.float64)
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)
        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])


#%%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset = AcuteDataset('dataset/dermal', classification = False, log_transform = True)

num_test = round(len(dataset) * 0.1)
num_val = round(len(dataset) * 0.1)
num_train = len(dataset) - (num_val + num_test)

seed = 0
np.random.seed(seed)
shuffled = np.random.permutation(len(dataset))

train_idx = shuffled[:num_train]
val_idx = shuffled[num_train:num_train+num_val]
test_idx = shuffled[-num_test:]

train_loader = DataLoader(dataset[train_idx], batch_size = num_train, shuffle = True, num_workers = 0)
val_loader = DataLoader(dataset[val_idx], batch_size = 32, shuffle = False, num_workers = 0)
test_loader = DataLoader(dataset[test_idx], batch_size = 32, shuffle = False, num_workers = 0)


#%%
def train(model, device, loader, criterion, optimizer):
    model.train()

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            
            def closure():
                optimizer.zero_grad()
                pred = model(batch)
                is_labeled = batch.y == batch.y
                loss = criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
                loss.backward(retain_graph = True)
                return loss
            
            optimizer.step(closure)


@torch.no_grad()
def evaluation(model, device, loader, criterion):
    model.eval()
    
    # y_true = []
    loss_list = []
    
    for _, batch in enumerate(loader):
        batch = batch.to(device)
        
        if batch.x.shape[0] == 1:
            pass
        else:
            pred = model(batch)
            
            y = batch.y.view(pred.shape).to(torch.float32)
            is_labeled = y == y
            # y_true.append(y[is_labeled].detach().cpu())
            
            loss = criterion(pred[is_labeled], y[is_labeled])
            loss_list.append(loss)
        
    # y_true = torch.cat(y_true, dim = 0).numpy()
    loss_list = torch.stack(loss_list)
    
    return sum(loss_list)/len(loss_list)


#%%
# esol_params = torch.load('esol.pth')['gnn_model_state_dict']
# freesolv_params = torch.load('freesolv.pth')['gnn_model_state_dict']
# lipo_params = torch.load('lipo.pth')['gnn_model_state_dict']

# param_names = esol_params.keys()
# pretrain_model_params = OrderedDict()

# for name in param_names:
#     OrderedDict[name] = (esol_params[name] + freesolv_params[name] + lipo_params[name])



#%%
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
# model.load_state_dict(pretrain_model_params)
model = model.to(device)

optimizer = optim.LBFGS(model.parameters(), max_iter = 20, history_size = 20)
# optimizer = optim.LBFGS(filter(lambda p: p.requires_grad, model.parameters()))
# optimizer = optim.Adam(model.parameters(), lr = lr)
criterion = nn.MSELoss()

best_epoch = 0
best_val_loss = 1e+10
best_test_loss = 0
for epoch in range(1, 100+1):
    print(f'=== epoch {epoch}')
    
    train(model, device, train_loader, criterion, optimizer)
    # aa_train(model, device, train_loader, criterion, optimizer)
    
    train_loss = evaluation(model, device, train_loader, criterion)
    val_loss = evaluation(model, device, val_loader, criterion)
    test_loss = evaluation(model, device, test_loader, criterion)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_test_loss = test_loss
        best_epoch = epoch
        
        gnn_model_params = model.state_dict()
        
    print(f'train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, test loss: {test_loss:.4f}')

path = 'oral.pth'
gnn_check_points = {'epoch': best_epoch,
                    'gnn_model_state_dict': gnn_model_params,
                    'path': path,
                    'optimizer_state_dict': optimizer.state_dict()}


#%%
model.eval()

test_pred = []
for _, batch in enumerate(test_loader):
    batch = batch.to(device)
    
    pred = model(batch)
    test_pred.append(pred)

test_pred = torch.cat(test_pred).view(-1).detach()
test_y = dataset[test_idx].y.view(-1)


# %%
l = torch.linspace(0, max(torch.cat([test_pred, test_y])), 100)

plt.figure(figsize = (7, 7))
plt.plot(l, l)
plt.scatter(test_pred, test_y)
plt.xlabel('pred')
plt.ylabel('true')
plt.show()
plt.close()


#%%
l = torch.linspace(0, torch.exp(max(torch.cat([test_pred, test_y]))), 100)

plt.figure(figsize = (7, 7))
plt.plot(l, l)
plt.scatter(torch.exp(test_pred), torch.exp(test_y))
plt.xlabel('pred')
plt.ylabel('true')
plt.show()
plt.close()


# %%
x = torch.rand(size = (100, 10))
y = torch.rand(size = (100, 1))

class LinearRegression(nn.Module):
  def __init__(self, p):
    super(LinearRegression, self).__init__()

    self.p = p
    self.lin1 = nn.Linear(p, 5)
    self.lin2 = nn.Linear(5, 1)

    self.batchnorm = nn.BatchNorm1d(5)

  def forward(self, x):
    h = self.batchnorm(self.lin1(x))
    output = self.lin2(h)
    return output

model = LinearRegression(x.size(1))
optimizer = optim.LBFGS(model.parameters())
criterion = nn.MSELoss()

def closure():
  optimizer.zero_grad()
  pred = model(x)
  loss = criterion(pred, y.view(pred.shape))
  loss.backward()
  return loss

optimizer.step(closure)


#%%
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 데이터 생성
x = torch.linspace(-10, 10, 100).view(-1, 5)
y = 2 * x + 3 + torch.randn(x.size()) * 2  # 노이즈 추가

# 모델 정의
class SimpleLinearRegression(nn.Module):
    def __init__(self):
        super(SimpleLinearRegression, self).__init__()
        # self.linear = nn.Linear(1, 3)  # 1D 입력, 1D 출력
        self.linear = nn.Sequential(
            nn.Linear(5, 3),
            nn.Linear(3, 1),
        )

    def forward(self, x):
        return self.linear(x)

model = SimpleLinearRegression()

# 손실 함수
criterion = nn.MSELoss()

# LBFGS 최적화기 설정
optimizer = optim.LBFGS(model.parameters(), lr=0.01, max_iter=20)

# 학습
def closure():
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    return loss

for epoch in range(100):
    optimizer.step(closure)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {closure().item()}")

# 결과 시각화
with torch.no_grad():
    plt.scatter(x, y, color="blue")
    plt.plot(x, model(x), color="red")
    plt.title("LBFGS Optimization")
    plt.show()
# %%
