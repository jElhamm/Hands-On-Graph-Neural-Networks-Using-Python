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
    
class GraphOperations:
    @staticmethod
    def graph_convolution(A, X, W, W_att):
        connections = np.where(A > 0)
        concat = np.concatenate([(X @ W.T)[connections[0]], (X @ W.T)[connections[1]]], axis=1)
        a = W_att @ concat.T
        e = GraphOperations.leaky_relu(a)
        E = np.zeros(A.shape)
        E[connections[0], connections[1]] = e[0]
        W_alpha = GraphOperations.softmax2D(E, 1)
        H = A.T @ W_alpha @ X @ W.T
        return H

    @staticmethod
    def leaky_relu(x, alpha=0.2):
        return np.maximum(alpha * x, x)

    @staticmethod
    def softmax2D(x, axis):
        e = np.exp(x - np.expand_dims(np.max(x, axis=axis), axis))
        sum = np.expand_dims(np.sum(e, axis=axis), axis)
        return e / sum
    
# ----------------------------------------------------------------------- GAT Model -----------------------------------------------------------------------------
    
class GAT(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, heads=8):
        super().__init__()
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
        self.gat2 = GATv2Conv(dim_h * heads, dim_out, heads=1)

    def forward(self, x, edge_index):
        h = F.dropout(x, p=0.6, training=self.training)
        h = self.gat1(h, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=0.6, training=self.training)
        h = self.gat2(h, edge_index)
        return F.log_softmax(h, dim=1)

    def fit(self, data, epochs):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=0.01)
        self.train()
        for epoch in range(epochs + 1):
            optimizer.zero_grad()
            out = self(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            acc = self.accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
                val_acc = self.accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask])
                print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: {acc*100:>5.2f}% | Val Loss: {val_loss:.2f} | Val Acc: {val_acc*100:.2f}%')
    
    @torch.no_grad()
    def test(self, data):
        self.eval()
        out = self(data.x, data.edge_index)
        acc = self.accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
        return acc

    @staticmethod
    def accuracy(y_pred, y_true):
        return torch.sum(y_pred == y_true) / len(y_true)
    
class DataVisualizer:
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
    def plot_accuracy_by_degree(degrees, accuracies, sizes):
        fig, ax = plt.subplots()
        ax.set_xlabel('Node degree')
        ax.set_ylabel('Accuracy score')
        plt.bar(['0', '1', '2', '3', '4', '5', '6+'], accuracies)
        for i in range(0, 7):
            plt.text(i, accuracies[i], f'{accuracies[i]*100:.2f}%', ha='center', color='black')
        for i in range(0, 7):
            plt.text(i, accuracies[i] // 2, sizes[i], ha='center', color='white')
        plt.show()
    

# --------------------------------------------------------------------------------------------------------------------------------------------------------------
        
SeedSetter.set_seeds()                                                                                 # Set seeds
dataset_cora = Planetoid(root=".", name="Cora")                                                        # Load Cora dataset
data_cora = dataset_cora[0]

# -------------------------------------------------------------- Graph Convolution Operation Example -----------------------------------------------------------

A = np.array([
    [1, 1, 1, 1],
    [1, 1, 0, 0],
    [1, 0, 1, 1],
    [1, 0, 1, 1]
  ])
X = np.random.uniform(-1, 1, (4, 4))
W = np.random.uniform(-1, 1, (2, 4))
W_att = np.random.uniform(-1, 1, (1, 4))
H = GraphOperations.graph_convolution(A, X, W, W_att)
print("Graph Convolution Output:\n", H)
    