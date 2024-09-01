# *******************************************************************************************************************************************
#                                                                                                                                           *
#                               Hands On Graph Neural Networks Using Python  -  CHAPTER 10                                                  *
#                                                                                                                                           *
#       This script sets up and runs experiments with two graph neural network models, VGAE and DGCNN, using the Cora dataset from          *
#       PyTorch Geometric. It first defines several classes: Encoder for the VGAE encoder network, VGAEModel for the VGAE model             *
#       including training and testing methods, SealProcessing for processing graph data with special transformations, and DGCNNModel       *
#       for a deep graph convolutional neural network with 1D convolutional layers. The GraphModeling class orchestrates data loading,      *
#       model training, and evaluation for both models. It initializes the dataset, processes it, and trains the VGAE model to predict      *
#       graph link probabilities, evaluating performance with metrics like AUC and average precision. It also trains the DGCNN model        *
#       for edge classification, with separate loaders for positive and negative edges.                                                     *
#       Finally, it prints out the results from both models' evaluations.                                                                   *
#                                                                                                                                           *
# *******************************************************************************************************************************************


# !pip install -q torch-scatter~=2.1.0 torch-sparse~=0.6.16 torch-cluster~=1.6.0 torch-spline-conv~=1.2.1 torch-geometric==2.2.0 -f https://data.pyg.org/whl/torch-{torch.__version__}.html

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from scipy.sparse.csgraph import shortest_path
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import GCNConv, VGAE, global_sort_pool
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import k_hop_subgraph, to_scipy_sparse_matrix
from torch.nn import Conv1d, MaxPool1d, Linear, Dropout, BCEWithLogitsLoss
import warnings
warnings.filterwarnings("ignore")
    