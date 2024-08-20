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
    