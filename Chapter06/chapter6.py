# ***************************************************************************************************************************************
#                                                                                                                                       *
#                               Hands On Graph Neural Networks Using Python  -  CHAPTER 6                                               *
#                                                                                                                                       *
#       This code performs various operations on graph data using PyTorch Geometric and other libraries.                                *
#       It starts by setting random seeds for reproducibility. It then demonstrates linear algebra operations,                          *
#       including matrix inversion. The script visualizes node degree distributions for two datasets:                                   *
#       Cora from the Planetoid dataset and the Facebook Page-Page dataset. A Graph Convolutional Network (GCN) is created and          *
#       trained for node classification on these datasets. Additionally, a GCN is extended for regression tasks and trained on          *
#       the Wikipedia Network dataset. The script also includes utilities for loading data, plotting distributions,                     *
#       and evaluating model performance with metrics such as Mean Squared Error (MSE) and Mean Absolute Error (MAE).                   *
#       Finally, it plots the regression results to compare ground truth versus predicted values.                                       *
#                                                                                                                                       *
# ***************************************************************************************************************************************



# !pip install -q torch-scatter~=2.1.0 torch-sparse~=0.6.16 torch-cluster~=1.6.0 torch-spline-conv~=1.2.1 torch-geometric==2.2.0 -f https://data.pyg.org/whl/torch-{torch.__version__}.html

import torch
import numpy as np
import pandas as pd
import seaborn as sns
from io import BytesIO
from zipfile import ZipFile
from scipy.stats import norm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from urllib.request import urlopen
from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree
from torch_geometric.transforms import RandomNodeSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch_geometric.datasets import Planetoid, FacebookPagePage, WikipediaNetwork
    

# --------------------------------------------------------- Set random seeds for reproducibility ----------------------------------------------------------------

class SeedSetter:
    @staticmethod
    def set_seeds(seed=1):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
    

# ---------------------------------------------------------------- Linear Algebra Operations --------------------------------------------------------------------
        
class LinearAlgebraOperations:
    @staticmethod
    def inverse_operations():
        D = np.array([
            [3, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 2]
        ])
        print(np.linalg.inv(D))
        print(np.linalg.inv(D + np.identity(4)))

        A = np.array([
            [1, 1, 1, 1],
            [1, 1, 0, 0],
            [1, 0, 1, 1],
            [1, 0, 1, 1]
        ])

        print(np.linalg.inv(D + np.identity(4)) @ A)
        print()
        print(A @ np.linalg.inv(D + np.identity(4)))
    

# -------------------------------------------------------------- Dataset and Data Visualization ---------------------------------------------------------------
        
class DatasetVisualizer:
    @staticmethod
    def visualize_planetoid():
        dataset = Planetoid(root=".", name="Cora")
        data = dataset[0]
        degrees = degree(data.edge_index[0]).numpy()
        numbers = Counter(degrees)

        fig, ax = plt.subplots()
        ax.set_xlabel('Node degree')
        ax.set_ylabel('Number of nodes')
        plt.bar(numbers.keys(), numbers.values())
        plt.show()

    @staticmethod
    def visualize_facebook_page():
        dataset = FacebookPagePage(root=".")
        data = dataset[0]
        data.train_mask = range(18000)
        data.val_mask = range(18001, 20000)
        data.test_mask = range(20001, 22470)

        degrees = degree(data.edge_index[0]).numpy()
        numbers = Counter(degrees)

        fig, ax = plt.subplots()
        ax.set_xlabel('Node degree')
        ax.set_ylabel('Number of nodes')
        plt.bar(numbers.keys(), numbers.values())
        plt.show()
    

# -------------------------------------------------------------- Graph Convolutional Network Class -------------------------------------------------------------
        
class GCN(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.gcn1 = GCNConv(dim_in, dim_h)
        self.gcn2 = GCNConv(dim_h, dim_out)

    def forward(self, x, edge_index):
        h = self.gcn1(x, edge_index)
        h = torch.relu(h)
        h = self.gcn2(h, edge_index)
        return F.log_softmax(h, dim=1)

    def fit(self, data, epochs):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        self.train()
        for epoch in range(epochs + 1):
            optimizer.zero_grad()
            out = self(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask].long())
            acc = (out[data.train_mask].argmax(dim=1) == data.y[data.train_mask].long()).float().mean()
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask].long())
                val_acc = (out[data.val_mask].argmax(dim=1) == data.y[data.val_mask].long()).float().mean()
                print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: {acc*100:>5.2f}% | Val Loss: {val_loss:.2f} | Val Acc: {val_acc*100:.2f}%')
    
    @torch.no_grad()
    def test(self, data):
        self.eval()
        out = self(data.x, data.edge_index)
        acc = (out.argmax(dim=1)[data.test_mask] == data.y[data.test_mask].long()).float().mean()
        return acc
    