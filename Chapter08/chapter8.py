# ***************************************************************************************************************************************
#                                                                                                                                       *
#                               Hands On Graph Neural Networks Using Python  -  CHAPTER 8                                               *
#                                                                                                                                       *
#       This code demonstrates a comprehensive workflow for loading, training, and evaluating a GraphSAGE model using PyTorch           *
#       Geometric on the PPI (Protein-Protein Interaction) dataset. It begins by setting a fixed random seed for reproducibility,       *
#       then defines a PPIDataLoader class for managing the loading and preparation of training, validation, and test datasets.         *
#       This class uses neighbor sampling to create data loaders tailored for the PPI dataset. The GraphSAGETrainer class is used       *
#       to initialize and manage the GraphSAGE model, including its training and evaluation. The model is trained over a specified      *
#       number of epochs, with training and validation metrics reported periodically. After training, the model's performance is        *
#       evaluated on the test set, with the final F1-score providing an assessment of its accuracy. The code integrates these           *
#       components to ensure a structured approach to handling graph data, training models, and evaluating performance.                 *
#                                                                                                                                       *
# ***************************************************************************************************************************************



# !pip install -q torch-scatter~=2.1.0 torch-sparse~=0.6.16 torch-cluster~=1.6.0 torch-spline-conv~=1.2.1 torch-geometric==2.2.0 -f https://data.pyg.org/whl/torch-{torch.__version__}.html

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.data import Batch
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import Planetoid, PPI
from torch_geometric.loader import NeighborLoader, DataLoader
from torch_geometric.nn import SAGEConv, GraphSAGE as PyGGraphSAGE
    

# ---------------------------------------------------------------- Set random seeds for reproducibility ----------------------------------------------------------------

class RandomSeedSetup:
    def __init__(self, seed=0):
        self.seed = seed
        self.setup()

    def setup(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    