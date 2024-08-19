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
    

# ------------------------------------------------------------------ Define MLP model ------------------------------------------------------------------------------
        
class MLP(torch.nn.Module): 
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.linear1 = Linear(dim_in, dim_h)
        self.linear2 = Linear(dim_h, dim_out)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return F.log_softmax(x, dim=1)

    def fit(self, data, epochs, learning_rate=0.01, weight_decay=5e-4):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.train()
        for epoch in range(epochs+1):
            optimizer.zero_grad()
            out = self(data.x)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            acc = self.accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
                val_acc = self.accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask])
                print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc:'
                      f' {acc*100:>5.2f}% | Val Loss: {val_loss:.2f} | '
                      f'Val Acc: {val_acc*100:.2f}%')

    @torch.no_grad()
    def test(self, data):
        self.eval()
        out = self(data.x)
        acc = self.accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
        return acc

    @staticmethod
    def accuracy(y_pred, y_true):
        return torch.sum(y_pred == y_true).item() / len(y_true)
    


# ------------------------------------------------------------- Define Vanilla GNN layer and model -----------------------------------------------------------------
    
class VanillaGNNLayer(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear = Linear(dim_in, dim_out, bias=False)

    def forward(self, x, adjacency):
        x = self.linear(x)
        x = torch.sparse.mm(adjacency, x)
        return x
    

# ----------------------------------------------- VanillaGNN model with two GNN layers for node classification------------------------------------------------------
    
class VanillaGNN(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.gnn1 = VanillaGNNLayer(dim_in, dim_h)
        self.gnn2 = VanillaGNNLayer(dim_h, dim_out)

    def forward(self, x, adjacency):
        h = self.gnn1(x, adjacency)
        h = torch.relu(h)
        h = self.gnn2(h, adjacency)
        return F.log_softmax(h, dim=1)
    
    def fit(self, data, adjacency, epochs, learning_rate=0.01, weight_decay=5e-4):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        self.train()
        for epoch in range(epochs+1):
            optimizer.zero_grad()
            out = self(data.x, adjacency)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            acc = MLP.accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
                val_acc = MLP.accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask])
                print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc:'
                      f' {acc*100:>5.2f}% | Val Loss: {val_loss:.2f} | '
                      f'Val Acc: {val_acc*100:.2f}%')

    @torch.no_grad()
    def test(self, data, adjacency):
        self.eval()
        out = self(data.x, adjacency)
        acc = MLP.accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
        return acc
    

# ------------------------------------------------------ Utility function for creating adjacency matrix -------------------------------------------------------------
    
class GraphUtils:
    @staticmethod
    def create_adjacency_matrix(data):
        adjacency = to_dense_adj(data.edge_index)[0]
        adjacency += torch.eye(len(adjacency))
        return adjacency
    

# ------------------------------------------------------------- Load and analyze Cora dataset ----------------------------------------------------------------------
    
cora_loader = DatasetLoader("Cora")
cora_loader.print_dataset_info()
cora_loader.print_graph_info()

# -------------------------------------------------------------- Convert data to DataFrame -------------------------------------------------------------------------

df_x = pd.DataFrame(cora_loader.data.x.numpy())
df_x['label'] = pd.DataFrame(cora_loader.data.y)
print(df_x.head())
     