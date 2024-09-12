# Chapter 12

   This chapter showcases the implementation and training of a `Heterogeneous Graph Attention Network (HAN)` using PyTorch Geometric. You’ll explore how to define and utilize different graph neural network layers like **GCNConv**, **GATModel**, and **HANModel** for node classification tasks using the **DBLP dataset**.

## Project Overview

   The code in this chapter focuses on building, training, and evaluating a Graph Neural Network (GNN) model with multiple key features:

   - Three key models:

      - GCNConv for Graph Convolutional Networks.

      - GATModel for Graph Attention Networks.

      - HANModel for Heterogeneous Graph Attention Networks, which is used for training.

   - Data is sourced from the DBLP dataset for academic networks.

   - Model evaluation: The training process includes monitoring accuracy, precision, recall, and F1 scores every 20 epochs.

   - Model checkpoints: The model is saved periodically during training.

   After training, the model is tested on a separate test set, and performance metrics are computed.

## Files Included

   This folder contains the following files:

   - [chapter12.py](Chapter12/chapter12.py): Python script with the entire code.

   - [chapter12.ipynb](Chapter12/chapter12.ipynb): Jupyter Notebook for a more interactive coding experience.

   Both files demonstrate the same code in different formats, allowing flexibility for those who prefer either a script or a notebook.

## Installation

   Before running the code, ensure the necessary dependencies are installed. The primary dependencies are PyTorch and PyTorch Geometric.

   To install them, run the following command:

```bash
   !pip install -q torch-scatter~=2.1.0 torch-sparse~=0.6.16 torch-cluster~=1.6.0 torch-spline-conv~=1.2.1 torch-geometric==2.2.0 -f https://data.pyg.org/whl/torch-{torch.__version__}.html
```

## Training and Evaluation

   - The training loop runs for 101 epochs.

   - Every 20 epochs, the training and validation performance are measured using accuracy, precision, recall, and F1-score.

   - The final test set evaluation occurs at the end of training, computing similar metrics on unseen data.