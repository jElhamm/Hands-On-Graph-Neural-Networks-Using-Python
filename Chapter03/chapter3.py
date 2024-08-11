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
    