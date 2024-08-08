#********************************************************************************************************************************************************
#                                            Hands On Graph Neural Networks Using Python  -  CHAPTER 2                                                  *
#                                                                                                                                                       *
#       - This code provides functionality for graph visualization and analysis using NetworkX and Matplotlib.                                          *
#       - The GraphVisualizer class includes methods to set a graph, draw a basic graph, and draw a weighted graph with edge labels.                    *
#       - The GraphAnalyzer class offers static methods to check graph connectivity and print centrality measures (degree, closeness,                   *
#         and betweenness centrality).                                                                                                                  *
#       - The SimpleGraphOperations class provides static methods to perform breadth-first search (BFS) and depth-first search (DFS) on a graph.        *
#       - Several types of graphs are defined: an undirected simple graph, a directed graph, and a weighted graph.                                      *
#         The graphs are visualized using the GraphVisualizer class.                                                                                    *
#       - Connectivity of two example graphs is checked using the GraphAnalyzer class.                                                                  *
#       - Centrality measures for the simple graph are printed.                                                                                         *
#       - BFS and DFS traversals are performed on the simple graph, and their results are printed.                                                      *
#       - Adjacency matrix and adjacency list representations of a graph are also demonstrated.                                                         *
#                                                                                                                                                       *
#********************************************************************************************************************************************************



import networkx as nx
import matplotlib.pyplot as plt


# ------------------------------------------------------- visualizing graphs using NetworkX and Matplotlib -----------------------------------------------------

class GraphVisualizer:
    def __init__(self):
        self.graph = None

    def set_graph(self, graph):
        self.graph = graph

    def draw_graph(self, layout='spring', node_size=600, cmap='coolwarm', font_size=14, font_color='white'):
        if self.graph is None:
            raise ValueError("Graph not set")

        plt.figure(figsize=(8, 6))
        plt.axis('off')
        pos = getattr(nx, f'{layout}_layout')(self.graph, seed=0)
        nx.draw_networkx(self.graph, pos=pos, node_size=node_size, cmap=cmap, font_size=font_size, font_color=font_color)
        plt.show()

    def draw_weighted_graph(self, weight_attr='weight'):
        if self.graph is None:
            raise ValueError("Graph not set")

        plt.figure(figsize=(8, 6))
        plt.axis('off')
        pos = nx.spring_layout(self.graph, seed=0)
        nx.draw_networkx(self.graph, pos=pos, node_size=600, cmap='coolwarm', font_size=14, font_color='white')
        labels = nx.get_edge_attributes(self.graph, weight_attr)
        nx.draw_networkx_edge_labels(self.graph, pos=pos, edge_labels=labels)
        plt.show()
    