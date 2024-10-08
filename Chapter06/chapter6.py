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
    

# --------------------------------------------------------- Set random seeds for reproducibility ----------------------------------------------------------------

class SeedSetter:
    @staticmethod
    def set_seeds(seed=1):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
    

# ---------------------------------------------------------------- Linear Algebra Operations --------------------------------------------------------------------
        
class LinearAlgebraOperations:
    @staticmethod
    def inverse_operations():
        D = np.array([
            [3, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 2]
        ])
        print(np.linalg.inv(D))
        print(np.linalg.inv(D + np.identity(4)))

        A = np.array([
            [1, 1, 1, 1],
            [1, 1, 0, 0],
            [1, 0, 1, 1],
            [1, 0, 1, 1]
        ])

        print(np.linalg.inv(D + np.identity(4)) @ A)
        print()
        print(A @ np.linalg.inv(D + np.identity(4)))
    

# -------------------------------------------------------------- Dataset and Data Visualization ---------------------------------------------------------------
        
class DatasetVisualizer:
    @staticmethod
    def visualize_planetoid():
        dataset = Planetoid(root=".", name="Cora")
        data = dataset[0]
        degrees = degree(data.edge_index[0]).numpy()
        numbers = Counter(degrees)

        fig, ax = plt.subplots()
        ax.set_xlabel('Node degree')
        ax.set_ylabel('Number of nodes')
        plt.bar(numbers.keys(), numbers.values())
        plt.show()

    @staticmethod
    def visualize_facebook_page():
        dataset = FacebookPagePage(root=".")
        data = dataset[0]
        data.train_mask = range(18000)
        data.val_mask = range(18001, 20000)
        data.test_mask = range(20001, 22470)

        degrees = degree(data.edge_index[0]).numpy()
        numbers = Counter(degrees)

        fig, ax = plt.subplots()
        ax.set_xlabel('Node degree')
        ax.set_ylabel('Number of nodes')
        plt.bar(numbers.keys(), numbers.values())
        plt.show()
    

# -------------------------------------------------------------- Graph Convolutional Network Class -------------------------------------------------------------
        
class GCN(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.gcn1 = GCNConv(dim_in, dim_h)
        self.gcn2 = GCNConv(dim_h, dim_out)

    def forward(self, x, edge_index):
        h = self.gcn1(x, edge_index)
        h = torch.relu(h)
        h = self.gcn2(h, edge_index)
        return F.log_softmax(h, dim=1)

    def fit(self, data, epochs):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        self.train()
        for epoch in range(epochs + 1):
            optimizer.zero_grad()
            out = self(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask].long())
            acc = (out[data.train_mask].argmax(dim=1) == data.y[data.train_mask].long()).float().mean()
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask].long())
                val_acc = (out[data.val_mask].argmax(dim=1) == data.y[data.val_mask].long()).float().mean()
                print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: {acc*100:>5.2f}% | Val Loss: {val_loss:.2f} | Val Acc: {val_acc*100:.2f}%')
    
    @torch.no_grad()
    def test(self, data):
        self.eval()
        out = self(data.x, data.edge_index)
        acc = (out.argmax(dim=1)[data.test_mask] == data.y[data.test_mask].long()).float().mean()
        return acc
    

# --------------------------------------------------------------- Extended GCN for Regression --------------------------------------------------------------------
    
class GCNRegressor(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.gcn1 = GCNConv(dim_in, dim_h * 4)
        self.gcn2 = GCNConv(dim_h * 4, dim_h * 2)
        self.gcn3 = GCNConv(dim_h * 2, dim_h)
        self.linear = torch.nn.Linear(dim_h, dim_out)

    def forward(self, x, edge_index):
        h = self.gcn1(x, edge_index)
        h = torch.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.gcn2(h, edge_index)
        h = torch.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.gcn3(h, edge_index)
        h = torch.relu(h)
        h = self.linear(h)
        return h
    
    def fit(self, data, epochs):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.02, weight_decay=5e-4)
        self.train()
        for epoch in range(epochs + 1):
            optimizer.zero_grad()
            out = self(data.x, data.edge_index)
            loss = F.mse_loss(out.squeeze()[data.train_mask], data.y[data.train_mask].float())
            loss.backward()
            optimizer.step()
            if epoch % 20 == 0:
                val_loss = F.mse_loss(out.squeeze()[data.val_mask], data.y[data.val_mask])
                print(f"Epoch {epoch:>3} | Train Loss: {loss:.5f} | Val Loss: {val_loss:.5f}")

    def test(self, data):
        self.eval()
        out = self(data.x, data.edge_index)
        return F.mse_loss(out.squeeze()[data.test_mask], data.y[data.test_mask].float())
    

# ---------------------------------------------------------------------- Utilities Class -------------------------------------------------------------------------
    
class Utils:
    @staticmethod
    def load_wikipedia_data():
        url = 'https://snap.stanford.edu/data/wikipedia.zip'
        with urlopen(url) as zurl:
            with ZipFile(BytesIO(zurl.read())) as zfile:
                zfile.extractall('.')
        df = pd.read_csv('wikipedia/chameleon/musae_chameleon_target.csv')
        values = np.log10(df['target'])
        return torch.tensor(values)

    @staticmethod
    def plot_degree_distribution(data):
        degrees = degree(data.edge_index[0]).numpy()
        numbers = Counter(degrees)
        fig, ax = plt.subplots()
        ax.set_xlabel('Node degree')
        ax.set_ylabel('Number of nodes')
        plt.bar(numbers.keys(), numbers.values())
        plt.show()

    @staticmethod
    def plot_target_distribution(df, values):
        df['target'] = values
        fig = sns.histplot(df['target'], kde=True, stat='density', linewidth=0)
        plt.show()
    

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
        
SeedSetter.set_seeds()                                                                              # Set seeds
LinearAlgebraOperations.inverse_operations()                                                        # Linear Algebra Operations
DatasetVisualizer.visualize_planetoid()                                                             # Planetoid Dataset Visualization
DatasetVisualizer.visualize_facebook_page()                                                         # Facebook Page Dataset Visualization
dataset = Planetoid(root=".", name="Cora")                                                          # Load Planetoid Dataset
data = dataset[0]

# ------------------------------------------------------------------ Create and Train GCN Model ----------------------------------------------------------------

gcn_classification = GCN(dataset.num_features, 16, dataset.num_classes)
print(gcn_classification)
gcn_classification.fit(data, epochs=100)
acc = gcn_classification.test(data)
print(f'\nGCN test accuracy: {acc*100:.2f}%\n')
    

# ---------------------------------------------------------------- Load Facebook Page-Page Dataset --------------------------------------------------------------

dataset = FacebookPagePage(root=".")
data = dataset[0]
data.train_mask = range(18000)
data.val_mask = range(18001, 20000)
data.test_mask = range(20001, 22470)

# -------------------------------------------------------------- Train GCN Model on Facebook Dataset ------------------------------------------------------------

gcn_classification = GCN(dataset.num_features, 16, dataset.num_classes)
print(gcn_classification)
gcn_classification.fit(data, epochs=100)
acc = gcn_classification.test(data)
print(f'\nGCN test accuracy: {acc*100:.2f}%\n')
    

# -----------------------------------------------------------------Load Wikipedia Network Dataset ---------------------------------------------------------------

dataset = WikipediaNetwork(root=".", name="chameleon", transform=RandomNodeSplit(num_val=200, num_test=500))
data = dataset[0]
data.y = Utils.load_wikipedia_data()

# -------------------------------------------------------------- Print Dataset and Graph Information ------------------------------------------------------------

print(f'Dataset: {dataset}')
print('-------------------')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of nodes: {data.x.shape[0]}')
print(f'Number of unique features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
print(f'\nGraph:')
print('------')
print(f'Edges are directed: {data.is_directed()}')
print(f'Graph has isolated nodes: {data.has_isolated_nodes()}')
print(f'Graph has loops: {data.has_self_loops()}')

# -------------------------------------------------------- Plot Degree Distribution and Target Distribution -----------------------------------------------------

Utils.plot_degree_distribution(data)
df = pd.read_csv('wikipedia/chameleon/musae_chameleon_target.csv')
values = np.log10(df['target'])
Utils.plot_target_distribution(df, values)
    

# ------------------------------------------------------------- Create and Train GCN Regressor Model ------------------------------------------------------------

gcn_regressor = GCNRegressor(dataset.num_features, 128, 1)
print(gcn_regressor)
gcn_regressor.fit(data, epochs=200)
loss = gcn_regressor.test(data)
print(f'\nGCN test loss: {loss:.5f}\n')

# ---------------------------------------------------------------------- Compute Metrics ------------------------------------------------------------------------

out = gcn_regressor(data.x, data.edge_index)
y_pred = out.squeeze()[data.test_mask].detach().numpy()
mse = mean_squared_error(data.y[data.test_mask], y_pred)
mae = mean_absolute_error(data.y[data.test_mask], y_pred)
print('=' * 43)
print(f'MSE = {mse:.4f} | RMSE = {np.sqrt(mse):.4f} | MAE = {mae:.4f}')
print('=' * 43)

# --------------------------------------------------------------------- Plot Regression Results ------------------------------------------------------------------

fig = sns.regplot(x=data.y[data.test_mask].numpy(), y=y_pred)
fig.set(xlabel='Ground truth', ylabel='Predicted values')
plt.show()