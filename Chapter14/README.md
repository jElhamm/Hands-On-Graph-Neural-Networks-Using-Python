# Chapter 14: Training GIN and GCN Models with PyTorch Geometric and Captum

   This chapter demonstrates the implementation, training, and evaluation of two Graph Neural Network (GNN) models: Graph Isomorphism Network (GIN) and Graph Convolutional Network (GCN). We also utilize GNNExplainer and Captum for model explainability, allowing us to visualize feature importance and understand model predictions.

## Overview

   This codebase includes two files:

   - [chapter14.py](Chapter14/chapter14.py): The Python script version of the chapter code.

   - [chapter14.ipynb](Chapter14/chapter14.ipynb): A Jupyter Notebook version for an interactive experience.
   
   
   The core components of this chapter's implementation include:

   - GINModel: A Graph Isomorphism Network with three GINConv layers followed by fully connected layers for node classification.

   - GCNModel: A Graph Convolutional Network with two GCNConv layers used for graph-based tasks.

   - DataPreparation: Handles data loading and splitting for training, validation, and testing.

   - ModelTrainer: Facilitates the training and evaluation process for both GIN and GCN models.

## Key Features

   1. GINModel:
   
   - A powerful GNN architecture with three layers of GINConv.
   - Uses global_add_pool to aggregate node features across layers.
   - Implemented with ReLU activations and BatchNorm1d for normalization.
   - Dropout is applied for regularization during training.

   2. GCNModel:

   - A simpler model that uses GCNConv layers for node feature propagation.
   - Dropout and ReLU are also employed for regularization and non-linearity.

   3. GNNExplainer:
   
   - Provides feature importance explanations for GIN by highlighting influential nodes and edges in the graph.

   4. Captum Integration:
   
   - Used to interpret the GCN model's predictions on the Twitch dataset via IntegratedGradients.

## Datasets Used

   - MUTAG: A popular dataset from the TUDataset library, consisting of chemical compounds classified into two categories.

   - Twitch: A social network dataset for predicting streamer features from the Twitch platform.

## Dependencies

   To run this code, the following packages are required:

```bash
   pip install torch-scatter~=2.1.0 torch-sparse~=0.6.16 torch-cluster~=1.6.0 torch-spline-conv~=1.2.1 torch-geometric~=2.0.4
   pip install captum==0.6.0
```

## Results

   - The GIN model is trained on the MUTAG dataset, and its performance is evaluated on validation and test sets.

   - The GCN model is trained on the Twitch EN dataset, and accuracy is printed after training.

## Visualization

   - GNNExplainer is used for explaining the GIN model on a sample graph.

   - Captum is used for feature attribution in the GCN model.