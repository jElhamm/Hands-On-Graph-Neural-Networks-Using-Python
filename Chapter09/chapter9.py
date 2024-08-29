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
    