# *****************************************************************************************************************************************
#                                                                                                                                         *
#                               Hands On Graph Neural Networks Using Python  -  CHAPTER 14                                                *
#                                                                                                                                         *
#       This code defines and trains two graph neural network (GNN) models: a GIN model and a GCN model,                                  *
#       using PyTorch Geometric and Captum. The DataPreparation class handles dataset loading and splitting for training,                 *
#       validation, and testing. The GINModel class implements a Graph Isomorphism Network with three GINConv layers followed             *
#       by linear layers for classification. The GCNModel class implements a Graph Convolutional Network with two GCNConv layers.         *
#       The ModelTrainer class manages the training and evaluation process, including computing loss and accuracy. After training         *
#       the GIN model, the GNNExplainer is used to visualize feature importance on a sample graph. The GCN model is also trained on       *
#       a Twitch dataset, and its accuracy is computed.                                                                                   *
#       The code includes dataset loading, model training, evaluation, and explanation with visualization.                                *
#                                                                                                                                         *
# *****************************************************************************************************************************************



# !pip install -q torch-scatter~=2.1.0 torch-sparse~=0.6.16 torch-cluster~=1.6.0 torch-spline-conv~=1.2.1 torch-geometric~=2.0.4 -f https://data.pyg.org/whl/torch-{torch.__version__}.html
# !pip install -q captum==0.6.0

import torch
import numpy as np
import networkx as nx
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.nn import to_captum
from captum.attr import IntegratedGradients
from captum.attr import IntegratedGradients
from torch_geometric.utils import to_networkx
from torch_geometric.loader import DataLoader
from torch_geometric.nn import Explainer, to_captum
from torch_geometric.transforms import ToUndirected
from torch_geometric.datasets import TUDataset, Twitch
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GINConv, GCNConv, global_add_pool, GNNExplainer


# ----------------------------------------- The class manages dataset loading and preparation for training and testing -----------------------------------------

class DataPreparation:
    def __init__(self, dataset_name='MUTAG', twitch=False):
        if twitch:
            self.dataset = Twitch('.', name=dataset_name)
            self.data = self.dataset[0]
        else:
            self.dataset = TUDataset(root='data/TUDataset', name=dataset_name).shuffle()
            self.train_dataset = self.dataset[:int(len(self.dataset)*0.8)]
            self.val_dataset = self.dataset[int(len(self.dataset)*0.8):int(len(self.dataset)*0.9)]
            self.test_dataset = self.dataset[int(len(self.dataset)*0.9):]
            self.train_loader = DataLoader(self.train_dataset, batch_size=64, shuffle=True)
            self.val_loader = DataLoader(self.val_dataset, batch_size=64, shuffle=True)
            self.test_loader = DataLoader(self.test_dataset, batch_size=64, shuffle=True)
    
# ------------------------------------------ The class defines a Graph Isomorphism Network with three GINConv layers -------------------------------------------

class GINModel(torch.nn.Module):
    def __init__(self, dim_h, dataset):
        super(GINModel, self).__init__()
        self.conv1 = GINConv(
            Sequential(Linear(dataset.num_node_features, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.lin1 = Linear(dim_h*3, dim_h*3)
        self.lin2 = Linear(dim_h*3, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)
        h = torch.cat((h1, h2, h3), dim=1)
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        return F.log_softmax(h, dim=1)
    
# ---------------------------------------- The class implements a Graph Convolutional Network with two GCNConv layers ----------------------------------------

class GCNModel(torch.nn.Module):
    def __init__(self, dim_h, dataset):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, dim_h)
        self.conv2 = GCNConv(dim_h, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.conv2(h, edge_index)
        return F.log_softmax(h, dim=1)
    
# ------------------------------------ The class facilitates the training and evaluation of graph neural network models ----------------------------------------

class ModelTrainer:
    def __init__(self, model, dataset, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataset = dataset
    
    def train(self, epochs, train_loader, val_loader):
        for epoch in range(epochs+1):
            self.model.train()
            total_loss = 0
            acc = 0
            val_loss = 0
            val_acc = 0
            for data in train_loader:
                self.optimizer.zero_grad()
                out = self.model(data.x, data.edge_index, data.batch)
                loss = self.criterion(out, data.y)
                total_loss += loss / len(train_loader)
                acc += self.accuracy(out.argmax(dim=1), data.y) / len(train_loader)
                loss.backward()
                self.optimizer.step()
                val_loss, val_acc = self.test(val_loader)
            if epoch % 20 == 0:
                print(f'Epoch {epoch:>3} | Train Loss: {total_loss:.2f} | Train Acc: {acc*100:>5.2f}% | Val Loss: {val_loss:.2f} | Val Acc: {val_acc*100:.2f}%')
    
    def test(self, loader):
        self.model.eval()
        loss = 0
        acc = 0
        for data in loader:
            out = self.model(data.x, data.edge_index, data.batch)
            loss += self.criterion(out, data.y) / len(loader)
            acc += self.accuracy(out.argmax(dim=1), data.y) / len(loader)
        return loss, acc

    def accuracy(self, pred_y, y):
        return ((pred_y == y).sum() / len(y)).item()
