# ***************************************************************************************************************************************
#                                                                                                                                       *
#                               Hands On Graph Neural Networks Using Python  -  CHAPTER 7                                               *
#                                                                                                                                       *
#                                                                                                                                       *
#     This code sets up and evaluates a Graph Attention Network (GAT) model using the Cora and CiteSeer datasets.                       *
#     It starts by installing necessary libraries and importing required modules.                                                       *
#                                                                                                                                       *
#     The SeedSetter class ensures reproducibility by setting random seeds. The GraphOperations class includes methods for              *
#     performing graph convolutions, using attention mechanisms, and applying activation functions.                                     *
#                                                                                                                                       *
#     The GAT class defines a GAT model with two layers of attention, a forward pass, and methods for training and testing.             *
#     The DataVisualizer class provides methods for visualizing node degree distributions and model accuracy based on node degrees      *
#                                                                                                                                       *
#     The code performs the following steps: it initializes and trains a GAT model on the Cora dataset, evaluates its performance,      *
#     and visualizes node degrees. It then does the same for the CiteSeer dataset, additionally plotting accuracy by node degree.       *
#                                                                                                                                       *
# ***************************************************************************************************************************************



# !pip install -q torch-scatter~=2.1.0 torch-sparse~=0.6.16 torch-cluster~=1.6.0 torch-spline-conv~=1.2.1 torch-geometric==2.2.0 -f https://data.pyg.org/whl/torch-{torch.__version__}.html

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import Counter
from torch.nn import Linear, Dropout
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import degree
from torch_geometric.datasets import Planetoid
    

class SeedSetter:
    @staticmethod
    def set_seeds(seed=1):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
    