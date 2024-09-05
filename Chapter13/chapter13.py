# ***********************************************************************************************************************************************
#                                                                                                                                               *
#                               Hands On Graph Neural Networks Using Python  -  CHAPTER 13                                                      *
#                                                                                                                                               *
#       This script demonstrates the implementation and evaluation of temporal Graph Neural Networks (GNNs) using PyTorch Geometric             *
#       Temporal on two datasets: WikiMaths and EnglandCovid. It includes data loaders for each dataset to prepare and visualize                *
#       the data. The script defines various GNN models (EvolveGCNH, EvolveGCNO, and MPNNLSTM) and a common Trainer class for                   *
#       training and evaluating these models. The training process involves fitting the model on the training data, evaluating                  *
#       performance using Mean Squared Error (MSE) on the test data, and plotting predictions and regression results. The code sets             *
#       a random seed for reproducibility, initializes, trains, and evaluates the models, and visualizes the results for both datasets.         *
#                                                                                                                                               *
# ***********************************************************************************************************************************************



# !pip install -q torch-scatter~=2.1.0 torch-sparse~=0.6.16 torch-cluster~=1.6.0 torch-spline-conv~=1.2.1 torch-geometric==2.2.0 -f https://data.pyg.org/whl/torch-{torch.__version__}.html
# !pip install -q torch-geometric-temporal==0.54.0

import torch
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import seaborn as sns
import torch.optim as optim
import matplotlib.pyplot as plt
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.nn.recurrent import EvolveGCNH, EvolveGCNO, MPNNLSTM
from torch_geometric_temporal.dataset import WikiMathsDatasetLoader, EnglandCovidDatasetLoader
    