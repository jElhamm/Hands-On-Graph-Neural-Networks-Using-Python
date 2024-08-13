# **********************************************************************************************************
#                                                                                                          *
#                   Hands On Graph Neural Networks Using Python  -  CHAPTER 3                              *
#                                                                                                          *
#            This project demonstrates various concepts in natural language processing (NLP)               *
#            and network analysis, including text preprocessing, word embedding generation                 *
#            with Word2Vec, random walks on graphs, and node classification in the Karate Club             *
#            graph. It includes the creation and visualization of an Erdos-Renyi graph,                    *
#            skipgram generation from text, and training of Word2Vec models on both text and               *
#            random walks. The project concludes with a TSNE visualization of node embeddings              *
#            and classification using a RandomForestClassifier.                                            *
#                                                                                                          *
# **********************************************************************************************************



import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


np.random.seed(0)
CONTEXT_SIZE = 2
text = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc eu sem
scelerisque, dictum eros aliquam, accumsan quam. Pellentesque tempus, lorem ut
semper fermentum, ante turpis accumsan ex, sit amet ultricies tortor erat quis
nulla. Nunc consectetur ligula sit amet purus porttitor, vel tempus tortor
scelerisque. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices
posuere cubilia curae; Quisque suscipit ligula nec faucibus accumsan. Duis
vulputate massa sit amet viverra hendrerit. Integer maximus quis sapien id
convallis. Donec elementum placerat ex laoreet gravida. Praesent quis enim
facilisis, bibendum est nec, pharetra ex. Etiam pharetra congue justo, eget
imperdiet diam varius non. Mauris dolor lectus, interdum in laoreet quis,
faucibus vitae velit. Donec lacinia dui eget maximus cursus. Class aptent taciti
sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Vivamus
tincidunt velit eget nisi ornare convallis. Pellentesque habitant morbi
tristique senectus et netus et malesuada fames ac turpis egestas. Donec
tristique ultrices tortor at accumsan.
""".split()


# --------------------------------------------- handle text processing tasks, including skipgram generation and vocabulary size calculation ---------------------------------------------

class TextProcessor:
    def __init__(self, text, context_size):
        self.text = text
        self.context_size = context_size
        self.vocab = set(text)
        self.vocab_size = len(self.vocab)
        self.skipgrams = self.generate_skipgrams()

    def generate_skipgrams(self):
        skipgrams = []
        for i in range(self.context_size, len(self.text) - self.context_size):
            context = [self.text[j] for j in np.arange(i - self.context_size, i + self.context_size + 1) if j != i]
            skipgrams.append((self.text[i], context))
        return skipgrams
    
    def display_skipgrams(self, num=2):
        print(self.skipgrams[:num])

    def display_vocab_size(self):
        print(f"Length of vocabulary = {self.vocab_size}")
    

# --------------------------------------------------------------- create and manage a Word2Vec model for word embeddings ----------------------------------------------------------------
        
class WordEmbeddingModel:
    def __init__(self, sentences, vector_size=10, window=2, sg=1, min_count=0, workers=1, seed=0):
        self.model = Word2Vec(sentences,
                              vector_size=vector_size,
                              window=window,
                              sg=sg,
                              min_count=min_count,
                              workers=workers,
                              seed=seed)

    def train_model(self, sentences, epochs=10):
        self.model.train(sentences, total_examples=self.model.corpus_count, epochs=epochs)
    
    def display_word_embedding(self, word_index=0):
        print('\nWord embedding =')
        print(self.model.wv[self.model.wv.index_to_key[word_index]])

    def display_embedding_shape(self):
        print(f'Shape of embedding matrix: {self.model.wv.vectors.shape}')
    

# ------------------------------------------------------------------- Class to visualize graphs using matplotlib ----------------------------------------------------------------------
        
class GraphVisualizer:
    @staticmethod
    def plot_graph(graph, title="Graph"):
        plt.figure()
        plt.axis('off')
        nx.draw_networkx(graph,
                         pos=nx.spring_layout(graph, seed=0),
                         node_size=600,
                         cmap='coolwarm',
                         font_size=14,
                         font_color='white'
                         )
        plt.title(title)
        plt.show()
    
    @staticmethod
    def plot_karate_club_graph(graph, labels):
        plt.figure(figsize=(12, 12))
        plt.axis('off')
        nx.draw_networkx(graph,
                         pos=nx.spring_layout(graph, seed=0),
                         node_color=labels,
                         node_size=800,
                         cmap='coolwarm',
                         font_size=14,
                         font_color='white'
                         )
        plt.title("Karate Club Graph")
        plt.show()
    

# -------------------------------------------------------------------- Class to perform random walks on a graph ---------------------------------------------------------------------
        
class RandomWalker:
    def __init__(self, graph):
        self.graph = graph
        self.walks = []

    def random_walk(self, start, length):
        walk = [str(start)]
        for _ in range(length):
            neighbors = list(self.graph.neighbors(start))
            next_node = np.random.choice(neighbors)
            walk.append(str(next_node))
            start = next_node
        return walk

    def generate_walks(self, num_walks_per_node=80, walk_length=10):
        self.walks = []
        for node in self.graph.nodes:
            for _ in range(num_walks_per_node):
                self.walks.append(self.random_walk(node, walk_length))
        return self.walks
    

# ------------------------------------------------- Create an Erdos-Renyi graph with 10 nodes and a probability of 0.3 for edge creation -------------------------------------------------
    
# Seed is set to 1 for reproducibility
graph = nx.erdos_renyi_graph(10, 0.3, seed=1)
GraphVisualizer.plot_graph(graph, "Erdos Renyi Graph")


# ---------------------------------------------------------------------------- Initialize the text processor ---------------------------------------------------------------------------- 

text_processor = TextProcessor(text, CONTEXT_SIZE)                                                                                 # Initialize the text processor with a given text and context size
text_processor.display_skipgrams()                                                                                                 # Display the skip-gram pairs generated from the text
text_processor.display_vocab_size()                                                                                                # Display the vocabulary size of the text


# ------------------------------------------------------------------------------- Initialize the  word  -------------------------------------------------------------------------------- 

embedding_model = WordEmbeddingModel([text])                                                                                       # Initialize a word embedding model with the given text
embedding_model.train_model([text])                                                                                                # Train the word embedding model using the provided text
embedding_model.display_word_embedding()                                                                                           # Display the word embeddings learned by the model
embedding_model.display_embedding_shape()                                                                                          # Display the shape of the word embeddings matrix
    

# ------------------------------------------------------------ Create a random walker and Create a Karate Club graph--------------------------------------------------------------------

walker = RandomWalker(graph)                                                                                                       # Create a random walker instance for the Erdos-Renyi graph
print("\nRandom walk starting from node 0:", walker.random_walk(0, 10))                                                            # Perform a random walk starting from node 0 for 10 steps and print the result

karate_club_graph = nx.karate_club_graph()                                                                                         # Create a Karate Club graph and extract labels based on club membership
karate_labels = [1 if karate_club_graph.nodes[node]['club'] == 'Officer' else 0 for node in karate_club_graph.nodes]
GraphVisualizer.plot_karate_club_graph(karate_club_graph, karate_labels)                                                           # Visualize the Karate Club graph with node colors indicating club membershi

karate_walker = RandomWalker(karate_club_graph)                                                                                    # Create a random walker instance for the Karate Club graph
walks = karate_walker.generate_walks()                                                                                             # Generate random walks from the Karate Club graph and print the first walk
print("\nFirst random walk:", walks[0])
    

# ----------------------------------------------------------------------- Initialize a word embedding model ----------------------------------------------------------------------------

walks_embedding_model = WordEmbeddingModel(walks, vector_size=100, window=10)                                                      # Initialize a word embedding model with the generated random walks
walks_embedding_model.model.build_vocab(walks)                                                                                     # Build vocabulary and train the word embedding model on the random walks
walks_embedding_model.train_model(walks, epochs=30)
walks_embedding_model.display_embedding_shape()                                                                                    # Display the shape of the word embeddings matrix
    

# --------------------------------------------------------------------- Print nodes, similarity t-SNE result ----------------------------------------------------------------------------

print('\nNodes that are the most similar to node 0:')                                                                              # Print nodes that are most similar to node 0 based on word embeddings
for similarity in walks_embedding_model.model.wv.most_similar(positive=['0']):
    print(f'   {similarity}')
print(f"\nSimilarity between node 0 and 4: {walks_embedding_model.model.wv.similarity('0', '4')}")                                 # Print similarity score between node 0 and node 4

nodes_wv = np.array([walks_embedding_model.model.wv.get_vector(str(i)) for i in range(len(walks_embedding_model.model.wv))])       # Convert node embeddings to a numpy array
tsne = TSNE(n_components=2, learning_rate='auto', init='pca', random_state=0)                                                      # Apply t-SNE to reduce the dimensionality of node embeddings for visualization
tsne_result = tsne.fit_transform(nodes_wv)
plt.figure(figsize=(6, 6))                                                                                                         # Plot the t-SNE result with nodes colored based on their Karate Club labels
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], s=100, c=karate_labels, cmap="coolwarm")
plt.title("TSNE Visualization")
plt.show()
    