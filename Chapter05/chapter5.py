# ***************************************************************************************************************************************
#                                                                                                                                       *
#                               Hands On Graph Neural Networks Using Python  -  CHAPTER 4                                               *
#                                                                                                                                       *
#       This code demonstrates the use of Multilayer Perceptron (MLP) and Vanilla Graph Neural Network                                  *
#       (GNN) models for node classification on graph datasets using PyTorch and PyTorch Geometric.                                     *
#       It starts by installing necessary libraries and setting seeds for reproducibility.                                              *
#       It then loads and analyzes two datasets, "Cora" and "FacebookPagePage", and provides methods to examine their properties.       *
#       The code defines an MLP and a basic GNN model, each with methods for training and testing.                                      *
#       Finally, it trains and evaluates both models on the datasets,printing performance metrics to compare their effectiveness.       *
#                                                                                                                                       *
# ***************************************************************************************************************************************



# !pip install -q torch-scatter~=2.1.0 torch-sparse~=0.6.16 torch-cluster~=1.6.0 torch-spline-conv~=1.2.1 torch-geometric==2.2.0 -f https://data.pyg.org/whl/torch-{torch.__version__}.html

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.datasets import Planetoid, FacebookPagePage
from torch_geometric.utils import to_dense_adj
import pandas as pd


# --------------------------------------------------------- Initialize random seeds for reproducibility ----------------------------------------------------------------

class SeedInitializer:
    @staticmethod
    def initialize_seed(seed=0):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Initialize seed
SeedInitializer.initialize_seed()
    

# --------------------------------------------------------------- Load and analyze datasets ---------------------------------------------------------------------------

class DatasetLoader:
    def __init__(self, dataset_name="Cora", root="."):
        if dataset_name == "Cora":
            self.dataset = Planetoid(root=root, name=dataset_name)
        elif dataset_name == "FacebookPagePage":
            self.dataset = FacebookPagePage(root=root)
        self.data = self.dataset[0]

    def print_dataset_info(self):
        print(f'Dataset: {self.dataset}')
        print('---------------')
        print(f'Number of graphs: {len(self.dataset)}')
        print(f'Number of nodes: {self.data.x.shape[0]}')
        print(f'Number of features: {self.dataset.num_features}')
        print(f'Number of classes: {self.dataset.num_classes}')

    def print_graph_info(self):
        print(f'\nGraph:')
        print('------')
        print(f'Edges are directed: {self.data.is_directed()}')
        print(f'Graph has isolated nodes: {self.data.has_isolated_nodes()}')
        print(f'Graph has loops: {self.data.has_self_loops()}')
    