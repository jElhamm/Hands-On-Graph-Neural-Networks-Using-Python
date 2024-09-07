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



# ------------------------------------------------------------------ WikiMaths Data Loader Class -----------------------------------------------------------------

class WikiMathsDataLoader:
    def __init__(self):
        self.dataset = WikiMathsDatasetLoader().get_dataset()
        self.train_dataset, self.test_dataset = temporal_signal_split(self.dataset, train_ratio=0.5)
        self.df = self._prepare_dataframe()

    def _prepare_dataframe(self):
        mean_cases = [snapshot.y.mean().item() for snapshot in self.dataset]
        std_cases = [snapshot.y.std().item() for snapshot in self.dataset]
        df = pd.DataFrame(mean_cases, columns=['mean'])
        df['std'] = pd.DataFrame(std_cases, columns=['std'])
        df['rolling'] = df['mean'].rolling(7).mean()
        return df

    def plot_data(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.df['mean'], 'k-', label='Mean')
        plt.plot(self.df['rolling'], 'g-', label='Moving average')
        plt.grid(linestyle=':')
        plt.fill_between(self.df.index, self.df['mean'] - self.df['std'], self.df['mean'] + self.df['std'], color='r', alpha=0.1)
        plt.axvline(x=360, color='b', linestyle='--')
        plt.text(360, 1.5, 'Train/test split', rotation=-90, color='b')
        plt.xlabel('Time (days)')
        plt.ylabel('Normalized number of visits')
        plt.legend(loc='upper right')
        plt.show()
    
# -------------------------------------------------------------- England Covid Data Loader Class -----------------------------------------------------------------
        
class EnglandCovidDataLoader:
    def __init__(self):
        self.dataset = EnglandCovidDatasetLoader().get_dataset(lags=14)
        self.train_dataset, self.test_dataset = temporal_signal_split(self.dataset, train_ratio=0.8)
        self.df = self._prepare_dataframe()

    def _prepare_dataframe(self):
        mean_cases = [snapshot.y.mean().item() for snapshot in self.dataset]
        std_cases = [snapshot.y.std().item() for snapshot in self.dataset]
        df = pd.DataFrame(mean_cases, columns=['mean'])
        df['std'] = pd.DataFrame(std_cases, columns=['std'])
        return df

    def plot_data(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.df['mean'], 'k-')
        plt.grid(linestyle=':')
        plt.fill_between(self.df.index, self.df['mean'] - self.df['std'], self.df['mean'] + self.df['std'], color='r', alpha=0.1)
        plt.axvline(x=38, color='b', linestyle='--', label='Train/test split')
        plt.text(38, 1, 'Train/test split', rotation=-90, color='b')
        plt.xlabel('Reports')
        plt.ylabel('Mean normalized number of cases')
        plt.legend(loc='upper right')
        plt.show()
    
# ----------------------------------------------------------- TemporalGNN model class with EvolveGCNH ------------------------------------------------------------

class TemporalGNNWikiMaths(nn.Module):
    def __init__(self, node_count, dim_in):
        super(TemporalGNNWikiMaths, self).__init__()
        self.recurrent = EvolveGCNH(node_count, dim_in)
        self.linear = nn.Linear(dim_in, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight).relu()
        h = self.linear(h)
        return h

# ----------------------------------------------------------- TemporalGNN model class with EvolveGCNO -----------------------------------------------------------

class TemporalGNNEnglandCovidEvolveGCNO(nn.Module):
    def __init__(self, dim_in):
        super(TemporalGNNEnglandCovidEvolveGCNO, self).__init__()
        self.recurrent = EvolveGCNO(dim_in, 1)
        self.linear = nn.Linear(dim_in, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight).relu()
        h = self.linear(h)
        return h
    
# ----------------------------------------------------------- TemporalGNN model class with MPNNLSTM -------------------------------------------------------------

class TemporalGNNEnglandCovidMPNNLSTM(nn.Module):
    def __init__(self, dim_in, dim_h, num_nodes):
        super(TemporalGNNEnglandCovidMPNNLSTM, self).__init__()
        self.recurrent = MPNNLSTM(dim_in, dim_h, num_nodes, 1, 0.5)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(2*dim_h + dim_in, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight).relu()
        h = self.dropout(h)
        h = self.linear(h).tanh()
        return h
    