# Chapter 10

   In Chapter 10 of the Hands-On Graph Neural Networks Using Python, we delve into the practical implementations of graph neural networks, focusing on two advanced models: **Variational Graph Autoencoder (VGAE)** and **Deep Graph Convolutional Neural Network (DGCNN)**. Using the Cora dataset from PyTorch Geometric, this chapter demonstrates the design, training, and evaluation of these models, showing how they can be applied to real-world tasks such as link prediction and edge classification.

## Files

   - [chapter10.py](Chapter10/chapter10.py): Contains the Python script for running experiments using VGAE and DGCNN.

   - [chapter10.ipynb](Chapter10/chapter10.ipynb): A Jupyter Notebook version of the script, providing step-by-step explanations and visualizations for better understanding.

## VGAE and DGCNN

   The models in this chapter aim to solve two different tasks:

   - **VGAE**: A model designed for link prediction, predicting the existence of edges between nodes in a graph.

   - **DGCNN**: A convolutional model designed for edge classification, classifying whether edges belong to a positive or negative class.

   Both models are trained and tested on the Cora dataset, and the results are evaluated using metrics like AUC (Area Under the ROC Curve) and average precision.

## Key Components

   - **VGAEModel**: Handles the VGAE model, including encoding, loss computation, and link prediction. The training process is focused on minimizing the reconstruction loss and KL-divergence.

   - **SealProcessing**: A utility class used to process graph data, generating subgraphs and performing node labeling based on distances.

   - **DGCNNModel**: Implements a Deep Graph Convolutional Neural Network, which combines GCN layers and 1D convolutions for edge classification. It uses the global sort pooling layer to handle variable graph sizes.

   - **GraphModeling**: The main class responsible for orchestrating the training and evaluation of both models. It loads the dataset, processes the data, and coordinates model execution.

## Running the Experiments

   To run the experiments, you can use either the chapter10.py script or the chapter10.ipynb notebook, depending on whether you prefer working in a Python environment or an interactive Jupyter notebook.


   The VGAE model will output link prediction results, while the DGCNN model will classify edges. The training progress is printed periodically to show the loss, AUC, and AP metrics.