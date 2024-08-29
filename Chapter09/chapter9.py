# *******************************************************************************************************************************************
#                                                                                                                                           *
#                               Hands On Graph Neural Networks Using Python  -  CHAPTER 9                                                   *
#                                                                                                                                           *
#       This code is a comprehensive pipeline for graph classification using graph neural networks (GNNs). It starts by loading             *
#       and preprocessing the PROTEINS dataset from the TUDataset collection, splitting it into training, validation, and test sets,        *
#       and creating data loaders. Two GNN models, Graph Convolutional Network (GCN) and Graph Isomorphism Network (GIN), are               *
#       implemented and trained separately. Each model includes multiple graph convolution layers followed by global pooling and            *
#       classification layers. The Trainer class is responsible for training these models and evaluating their performance,                 *
#       calculating loss and accuracy metrics. The Visualization class plots the classification results for both models,                    *
#       highlighting correct and incorrect predictions on sample graphs. Finally, the EnsembleEvaluator class assesses the                  *
#       performance of an ensemble model that combines the predictions of GCN and GIN, providing a comparative accuracy evaluation.         *
#       This pipeline offers a complete workflow for training, evaluating, and visualizing GNN models on graph classification tasks.        *
#                                                                                                                                           *
# *******************************************************************************************************************************************



# !pip install -q torch-scatter~=2.1.0 torch-sparse~=0.6.16 torch-cluster~=1.6.0 torch-spline-conv~=1.2.1 torch-geometric==2.2.0 -f https://data.pyg.org/whl/torch-{torch.__version__}.html

import torch
import numpy as np
import networkx as nx
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import TUDataset
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool, global_add_pool
    

# ----------------------------------------------------------------- Set up reproducibility --------------------------------------------------------------------

torch.manual_seed(11)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ------------------------------------------------- Handles the loading and preprocessing of the dataset ------------------------------------------------------

class GraphDataset:
    def __init__(self, dataset_name='PROTEINS', root='.'):
        self.dataset = TUDataset(root=root, name=dataset_name).shuffle()
        self.num_classes = self.dataset.num_classes
        self.num_features = self.dataset.num_features

    def print_dataset_info(self):
        print(f'Dataset: {self.dataset}')
        print('-----------------------')
        print(f'Number of graphs: {len(self.dataset)}')
        print(f'Number of nodes: {self.dataset[0].x.shape[0]}')
        print(f'Number of features: {self.num_features}')
        print(f'Number of classes: {self.num_classes}')
    
    def split_dataset(self):
        total_size = len(self.dataset)
        train_size = int(total_size * 0.8)
        val_size = int(total_size * 0.1)
        test_size = total_size - train_size - val_size
        self.train_dataset = self.dataset[:train_size]
        self.val_dataset = self.dataset[train_size:train_size + val_size]
        self.test_dataset = self.dataset[train_size + val_size:]
        print(f'Training set   = {len(self.train_dataset)} graphs')
        print(f'Validation set = {len(self.val_dataset)} graphs')
        print(f'Test set       = {len(self.test_dataset)} graphs')

    def create_loaders(self, batch_size=64):
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True)
    
# ---------------------------------------------------------- Graph Convolutional Network (GCN) model ----------------------------------------------------------
        
class GCN(torch.nn.Module):
    def __init__(self, dim_h, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, dim_h)
        self.conv2 = GCNConv(dim_h, dim_h)
        self.conv3 = GCNConv(dim_h, dim_h)
        self.lin = Linear(dim_h, num_classes)

    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()
        h = self.conv3(h, edge_index)
        hG = global_mean_pool(h, batch)
        h = F.dropout(hG, p=0.5, training=self.training)
        return F.log_softmax(self.lin(h), dim=1)
    
# ----------------------------------------------------------- Graph Isomorphism Network (GIN) model -----------------------------------------------------------
    
class GIN(torch.nn.Module):
    def __init__(self, dim_h, num_features, num_classes):
        super(GIN, self).__init__()
        self.conv1 = GINConv(Sequential(Linear(num_features, dim_h),
                                        BatchNorm1d(dim_h),
                                        ReLU(),
                                        Linear(dim_h, dim_h),
                                        ReLU()))
        self.conv2 = GINConv(Sequential(Linear(dim_h, dim_h),
                                        BatchNorm1d(dim_h),
                                        ReLU(),
                                        Linear(dim_h, dim_h),
                                        ReLU()))
        self.conv3 = GINConv(Sequential(Linear(dim_h, dim_h),
                                        BatchNorm1d(dim_h),
                                        ReLU(),
                                        Linear(dim_h, dim_h),
                                        ReLU()))
        self.lin1 = Linear(dim_h * 3, dim_h * 3)
        self.lin2 = Linear(dim_h * 3, num_classes)

    def forward(self, x, edge_index, batch):
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)
        h = torch.cat((h1, h2, h3), dim=1)
        h = self.lin1(h).relu()
        h = F.dropout(h, p=0.5, training=self.training)
        return F.log_softmax(self.lin2(h), dim=1)
    