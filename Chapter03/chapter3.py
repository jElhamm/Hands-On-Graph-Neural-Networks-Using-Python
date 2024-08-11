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
    