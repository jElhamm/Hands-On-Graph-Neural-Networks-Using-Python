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

    def plot_graph(self):
        rows, cols = np.where(self.adj > 0)
        edges = zip(rows.tolist(), cols.tolist())
        G = nx.Graph()
        G.add_edges_from(edges)
        plt.figure(figsize=(10, 5))
        nx.draw(G, with_labels=True)
        plt.show()
    