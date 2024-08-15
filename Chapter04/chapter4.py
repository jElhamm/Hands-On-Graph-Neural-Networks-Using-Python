# ***************************************************************************************************************************
#                                                                                                                           *
#                               Hands On Graph Neural Networks Using Python  -  CHAPTER 4                                   *
#                                                                                                                           *
#       This script covers various tasks related to graph processing and recommendation systems, including:                 *
#                                                                                                                           *
#       1. Graph Creation and Visualization:                                                                                *
#           - Creates a random graph using the Erdos-Renyi model.                                                           *
#           - Visualizes the graph using network plotting tools.                                                            *
#                                                                                                                           *
#       2. Random Walk and Node2Vec:                                                                                        *
#           - Performs random walks on a graph to generate node sequences.                                                  *
#           - Applies `Node2Vec` to learn node embeddings from these sequences.                                             *
#           - Trains a `Word2Ve`c model on the generated walks.                                                             *
#           - Evaluates the quality of the learned embeddings using a Random Forest classifier.                             *
#                                                                                                                           *
#       3. MovieLens Data Processing and Recommendation:                                                                    *
#           - Downloads and extracts MovieLens dataset.                                                                     *
#           - Constructs a movie co-occurrence graph from user ratings.                                                     *
#           - Uses Node2Vec to learn embeddings for movies and recommends similar movies based on these embeddings.         *
#                                                                                                                           *
#       The script integrates graph analysis techniques with practical applications in recommendation systems.              *
#                                                                                                                           *
# ***************************************************************************************************************************



# !pip install -q node2vec==0.4.6
# !pip install -qI gensim==4.3.0
import random
import numpy as np
import pandas as pd
import networkx as nx
from io import BytesIO
from zipfile import ZipFile
from node2vec import Node2Vec
import matplotlib.pyplot as plt
from urllib.request import urlopen
from collections import defaultdict
from sklearn.metrics import accuracy_score
from gensim.models.word2vec import Word2Vec
from sklearn.ensemble import RandomForestClassifier
    


class GraphVisualization:
    @staticmethod
    def create_graph(n=10, p=0.3, seed=1):
        G = nx.erdos_renyi_graph(n, p, seed=seed, directed=False)
        return G

    @staticmethod
    def plot_graph(G):
        plt.figure()
        plt.axis('off')
        nx.draw_networkx(G,
                         pos=nx.spring_layout(G, seed=0),
                         node_size=600,
                         cmap='coolwarm',
                         font_size=14,
                         font_color='white')
        plt.show()
    
class RandomWalk:
    def __init__(self, graph, p=1, q=1):
        self.graph = graph
        self.p = p
        self.q = q

    def next_node(self, previous, current):
        neighbors = list(self.graph.neighbors(current))
        alphas = []
        for neighbor in neighbors:
            if neighbor == previous:
                alpha = 1/self.p
            elif self.graph.has_edge(neighbor, previous):
                alpha = 1
            else:
                alpha = 1/self.q
            alphas.append(alpha)
        probs = [alpha / sum(alphas) for alpha in alphas]
        return np.random.choice(neighbors, size=1, p=probs)[0]

    def generate_walk(self, start, length):
        walk = [start]
        for i in range(length):
            current = walk[-1]
            previous = walk[-2] if len(walk) > 1 else None
            next_node = self.next_node(previous, current)
            walk.append(next_node)
        return walk
    