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
    