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
    