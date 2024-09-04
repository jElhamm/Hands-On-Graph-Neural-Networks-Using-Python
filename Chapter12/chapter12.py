# ******************************************************************************************************************************************
#                                                                                                                                          *
#                               Hands On Graph Neural Networks Using Python  -  CHAPTER 12                                                 *
#                                                                                                                                          *
#       This code implements and trains a Graph Neural Network (GNN) model using PyTorch Geometric.                                        *
#       It includes three model classes:GCNConv, GATModel, and HANModel, with HANModel being used in the training loop. The dataset        *
#       is loaded from the DBLP collection, and a HANModel is initialized and trained for 101 epochs. During training, the model's         *
#       performance is evaluated using accuracy, precision, recall, and F1 score on both training and validation sets every 20 epochs.     *
#       Additionally, model checkpoints are saved periodically. After training, the model is tested on a separate test set,                *
#       and final metrics are computed. The training loss is plotted against epochs to visualize the learning process.                     *
#       The compute_metrics function calculates precision, recall, and F1 score from scikit-learn.                                         *
#                                                                                                                                          *
# ******************************************************************************************************************************************


# !pip install -q torch-scatter~=2.1.0 torch-sparse~=0.6.16 torch-cluster~=1.6.0 torch-spline-conv~=1.2.1 torch-geometric==2.2.0 -f https://data.pyg.org/whl/torch-{torch.__version__}.html

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.nn import to_hetero
from torch_geometric.datasets import DBLP
from torch_geometric.data import HeteroData
from torch_geometric.utils import add_self_loops, degree
from sklearn.metrics import precision_recall_fscore_support
from torch_geometric.nn import MessagePassing, GATConv, HANConv
import warnings
warnings.filterwarnings("ignore")
    