# *******************************************************************************************************************************
#                                                                                                                               *
#                               Hands On Graph Neural Networks Using Python  -  CHAPTER 15                                      *
#                                                                                                                               *
#       This Python script implements a temporal graph neural network (GNN) model to predict traffic speeds using               *
#       spatio-temporal data. It involves three main components: data preparation, model training, and evaluation.              *
#       First, the DataPreparation class loads traffic speed and distance data, computes an adjacency matrix based              *
#       on the distances between sensors, and plots various visualizations. The TemporalGNN class defines a temporal            *
#       GNN model using the A3TGCN layer to process time series data from multiple sensors. The ModelTraining class             *
#       handles model training and evaluation, computing performance metrics like MAE, RMSE, and MAPE. The script's             *
#       main function prepares the data, sets up the model, and trains it using a dataset created from the traffic data.        *
#       After training, the model is evaluated to predict future traffic speeds based on historical data.                       *
#                                                                                                                               *
# *******************************************************************************************************************************


# !pip install -q torch-scatter~=2.1.0 torch-sparse~=0.6.16 torch-cluster~=1.6.0 torch-spline-conv~=1.2.1 torch-geometric==2.2.0 -f https://data.pyg.org/whl/torch-{torch.__version__}.html
# !pip install -q torch-geometric-temporal==0.54.0

import torch
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric_temporal.nn.recurrent import A3TGCN
from torch_geometric_temporal.signal import StaticGraphTemporalSignal, temporal_signal_split


# ---------------------------------- responsible for loading and preparing traffic data from CSV files ---------------------------------------

class DataPreparation:
    def __init__(self, speed_file, distance_file):
        self.speeds = pd.read_csv(speed_file, names=range(0, 228))
        self.distances = pd.read_csv(distance_file, names=range(0, 228))
        self.adj = None

    def plot_speed_data(self):
        plt.figure(figsize=(10, 5), dpi=100)
        plt.plot(self.speeds)
        plt.grid(linestyle=':')
        plt.xlabel('Time (5 min)')
        plt.ylabel('Traffic speed')
        plt.show()

    def plot_mean_std_speed(self):
        mean = self.speeds.mean(axis=1)
        std = self.speeds.std(axis=1)
        plt.figure(figsize=(10, 5), dpi=100)
        plt.plot(mean, 'k-')
        plt.grid(linestyle=':')
        plt.fill_between(mean.index, mean - std, mean + std, color='r', alpha=0.1)
        plt.xlabel('Time (5 min)')
        plt.ylabel('Traffic speed')
        plt.show()

    def plot_correlation_matrices(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8))
        fig.tight_layout(pad=3.0)
        ax1.matshow(self.distances)
        ax1.set_xlabel("Sensor station")
        ax1.set_ylabel("Sensor station")
        ax1.title.set_text("Distance matrix")
        ax2.matshow(-np.corrcoef(self.speeds.T))
        ax2.set_xlabel("Sensor station")
        ax2.set_ylabel("Sensor station")
        ax2.title.set_text("Correlation matrix")
        plt.show()

    def compute_adj(self, sigma2=0.1, epsilon=0.5):
        d = self.distances.to_numpy() / 10000.0
        d2 = d * d
        n = self.distances.shape[0]
        w_mask = np.ones([n, n]) - np.identity(n)
        self.adj = np.exp(-d2 / sigma2) * (np.exp(-d2 / sigma2) >= epsilon) * w_mask

    def plot_adj_matrix(self):
        plt.figure(figsize=(8, 8))
        plt.matshow(self.adj, False)
        plt.colorbar()
        plt.xlabel("Sensor station")
        plt.ylabel("Sensor station")
        plt.show()
    
    def plot_graph(self):
        rows, cols = np.where(self.adj > 0)
        edges = zip(rows.tolist(), cols.tolist())
        G = nx.Graph()
        G.add_edges_from(edges)
        plt.figure(figsize=(10, 5))
        nx.draw(G, with_labels=True)
        plt.show()
    
# ---------------------------------- defines a temporal graph neural network model using the A3TGCN layer ------------------------------------
        
class TemporalGNN(torch.nn.Module):
    def __init__(self, dim_in, periods):
        super().__init__()
        self.tgnn = A3TGCN(in_channels=dim_in, out_channels=32, periods=periods)
        self.linear = torch.nn.Linear(32, periods)

    def forward(self, x, edge_index, edge_attr):
        h = self.tgnn(x, edge_index, edge_attr).relu()
        h = self.linear(h)
        return h
    
# ------------------------------------ manages the training and evaluation of the temporal GNN model -----------------------------------------
    
class ModelTraining:
    def __init__(self, model, train_dataset, test_dataset, optimizer, lags, speeds):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.optimizer = optimizer
        self.lags = lags
        self.speeds = speeds

    def train(self, epochs=30):
        self.model.train()
        for epoch in range(epochs):
            loss = 0
            step = 0
            for snapshot in self.train_dataset:
                y_pred = self.model(snapshot.x.unsqueeze(2), snapshot.edge_index, snapshot.edge_attr)
                loss += torch.mean((y_pred - snapshot.y) ** 2)
                step += 1
            loss = loss / (step + 1)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if epoch % 5 == 0:
                print(f"Epoch {epoch:>2} | Train MSE: {loss:.4f}")

    def evaluate(self):
        self.model.eval()
        y_test = []
        gnn_pred = []
        for snapshot in self.test_dataset:
            y_hat = snapshot.y.numpy()
            y_hat = self.inverse_zscore(y_hat, self.speeds.mean(axis=0), self.speeds.std(axis=0))
            y_test = np.append(y_test, y_hat)

            y_hat = self.model(snapshot.x.unsqueeze(2), snapshot.edge_index, snapshot.edge_weight).squeeze().detach().numpy()
            y_hat = self.inverse_zscore(y_hat, self.speeds.mean(axis=0), self.speeds.std(axis=0))
            gnn_pred = np.append(gnn_pred, y_hat)

        print(f'GNN MAE  = {self.MAE(gnn_pred, y_test):.4f}')
        print(f'GNN RMSE = {self.RMSE(gnn_pred, y_test):.4f}')
        print(f'GNN MAPE = {self.MAPE(gnn_pred, y_test):.4f}')

    @staticmethod
    def inverse_zscore(x, mean, std):
        return x * std + mean

    @staticmethod
    def MAE(real, pred):
        return np.mean(np.abs(pred - real))

    @staticmethod
    def RMSE(real, pred):
        return np.sqrt(np.mean((pred - real) ** 2))

    @staticmethod
    def MAPE(real, pred):
        return np.mean(np.abs(pred - real) / (real + 1e-5))
    

# --------------------------------------------------------------- Data preparation -----------------------------------------------------------
    
data_prep = DataPreparation('PeMSD7_V_228.csv', 'PeMSD7_W_228.csv')
data_prep.plot_speed_data()
data_prep.plot_mean_std_speed()
data_prep.plot_correlation_matrices()
data_prep.compute_adj()
data_prep.plot_adj_matrix()
data_prep.plot_graph()

# -------------------------------------------------------------- Create dataset --------------------------------------------------------------

speeds_norm = (data_prep.speeds - data_prep.speeds.mean(axis=0)) / data_prep.speeds.std(axis=0)
lags = 24
horizon = 48
xs = []
ys = []
for i in range(lags, speeds_norm.shape[0] - horizon):
    xs.append(speeds_norm.to_numpy()[i - lags:i].T)
    ys.append(speeds_norm.to_numpy()[i + horizon - 1])

edge_index = (np.array(data_prep.adj) > 0).nonzero()
dataset = StaticGraphTemporalSignal(edge_index, data_prep.adj[data_prep.adj > 0], xs, ys)
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)
    
# -------------------------------------------------------------- Model setup -----------------------------------------------------------------

model = TemporalGNN(lags, 1).to('cpu')
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
print(model)

# -------------------------------------------------------- Train and Evaluate the model ------------------------------------------------------

trainer = ModelTraining(model, train_dataset, test_dataset, optimizer, lags, data_prep.speeds)
trainer.train(epochs=30)

trainer.evaluate()