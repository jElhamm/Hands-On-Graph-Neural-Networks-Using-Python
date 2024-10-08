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
    
class Node2VecModel:
    def __init__(self, graph):
        self.graph = graph

    def create_walks(self, num_walks=80, walk_length=10, p=3, q=2):
        walker = RandomWalk(self.graph, p, q)
        walks = []
        for node in self.graph.nodes:
            for _ in range(num_walks):
                walks.append(walker.generate_walk(node, walk_length))
        return walks
    
    def train(self, walks, vector_size=100, window=10, workers=2, epochs=30, seed=0):
        model = Word2Vec(walks,
                         hs=1,
                         sg=1,
                         vector_size=vector_size,
                         window=window,
                         workers=workers,
                         min_count=1,
                         seed=seed)
        model.train(walks, total_examples=model.corpus_count, epochs=epochs, report_delay=1)
        return model
    
class Classifier:
    def __init__(self, model):
        self.model = model

    def train_and_evaluate(self, labels, train_mask, test_mask):
        clf = RandomForestClassifier(random_state=0)
        clf.fit(self.model.wv[train_mask], labels[train_mask])

        y_pred = clf.predict(self.model.wv[test_mask])
        acc = accuracy_score(y_pred, labels[test_mask])
        print(f'Node2Vec accuracy = {acc*100:.2f}%')
    
class MovieLensGraph:
    @staticmethod
    def download_and_extract(url, extract_to='.'):
        with urlopen(url) as zurl:
            with ZipFile(BytesIO(zurl.read())) as zfile:
                zfile.extractall(extract_to)

    @staticmethod
    def load_data():
        ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'unix_timestamp'])
        movies = pd.read_csv('ml-100k/u.item', sep='|', usecols=range(2), names=['movie_id', 'title'], encoding='latin-1')
        return ratings, movies

    @staticmethod
    def filter_high_ratings(ratings, threshold=4):
        return ratings[ratings.rating >= threshold]
    
    @staticmethod
    def create_graph_from_ratings(ratings, threshold=20):
        pairs = defaultdict(int)
        for group in ratings.groupby("user_id"):
            user_movies = list(group[1]["movie_id"])
            for i in range(len(user_movies)):
                for j in range(i+1, len(user_movies)):
                    pairs[(user_movies[i], user_movies[j])] += 1

        G = nx.Graph()
        for pair in pairs:
            movie1, movie2 = pair
            score = pairs[pair]
            if score >= threshold:
                G.add_edge(movie1, movie2, weight=score)

        print("Total number of graph nodes:", G.number_of_nodes())
        print("Total number of graph edges:", G.number_of_edges())
        return G
    
class MovieRecommender:
    def __init__(self, model, movies):
        self.model = model
        self.movies = movies

    def recommend(self, movie_title):
        movie_id = str(self.movies[self.movies.title == movie_title].movie_id.values[0])
        for id, similarity in self.model.wv.most_similar(movie_id)[:5]:
            title = self.movies[self.movies.movie_id == int(id)].title.values[0]
            print(f'{title}: {similarity:.2f}')
    

# -------------------------------------------------------------------- Graph Visualization --------------------------------------------------------------------
            
gv = GraphVisualization()
G = gv.create_graph()
gv.plot_graph(G)


# ----------------------------------------------------------------- Random Walk and Node2Vec -------------------------------------------------------------------

G = nx.karate_club_graph()
rw = RandomWalk(G, p=1, q=1)
n2v_model = Node2VecModel(G)
walks = n2v_model.create_walks()
model = n2v_model.train(walks)

labels = np.array([1 if G.nodes[node]['club'] == 'Officer' else 0 for node in G.nodes])
train_mask = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
test_mask = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33]
classifier = Classifier(model)
classifier.train_and_evaluate(labels, train_mask, test_mask)
    

# -------------------------------------------------------------------------------------------------------------------------------------------------------------

ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'unix_timestamp'])
ratings

movies = pd.read_csv('ml-100k/u.item', sep='|', usecols=range(2), names=['movie_id', 'title'], encoding='latin-1')
movies

ratings = ratings[ratings.rating >= 4]
ratings


# ------------------------------------------------------------------ MovieLens Recommender ---------------------------------------------------------------------

ml = MovieLensGraph()
ml.download_and_extract('https://files.grouplens.org/datasets/movielens/ml-100k.zip')
ratings, movies = ml.load_data()
filtered_ratings = ml.filter_high_ratings(ratings)
G = ml.create_graph_from_ratings(filtered_ratings)
node2vec = Node2Vec(G, dimensions=64, walk_length=20, num_walks=200, p=2, q=1, workers=1)
model = node2vec.fit(window=10, min_count=1, batch_words=4)

recommender = MovieRecommender(model, movies)
recommender.recommend('Star Wars (1977)')