## Chapter 9: Graph Classification with GCN and GIN

   In this chapter, we focus on graph classification using two powerful Graph Neural Networks: the **Graph Convolutional Network (GCN)** and the **Graph Isomorphism Network (GIN)**. The code provides a full pipeline for loading, preprocessing, training, and evaluating models on the PROTEINS dataset, widely used for graph-based machine learning tasks.

## Overview

   This chapter walks you through:

   - Loading and preprocessing the PROTEINS dataset from the TUDataset collection.

   - Implementing two GNN models: GCN and GIN.

   - Training and evaluating these models.

   - Visualizing the classification performance.

   - Using an ensemble model to combine GCN and GIN predictions for improved accuracy.

## Files

   - [chapter9.py](Chapter09/chapter09.py): Contains the full pipeline in Python script format.

   - [chapter9.ipynb](Chapter09/chapter09.ipynb): An interactive notebook version of the same code, allowing for step-by-step execution.

## Key Components

   **1. Dataset Preparation:**

   - We use the PROTEINS dataset for graph classification.

   - The dataset is split into training, validation, and test sets, and data loaders are created for efficient batch processing.

   **2. GCN and GIN Models:**

   - The GCN model employs three graph convolutional layers, followed by global mean pooling and a classification layer.

   - The GIN model uses a more expressive graph structure learning technique, leveraging three GIN convolutional layers followed by a classification layer.

   **3. Trainer Class:**

   - A reusable Trainer class handles the training and evaluation process.

   - It calculates the loss and accuracy at each epoch and prints the performance at intervals.

   **4. Visualization:**

   - A Visualization class is provided to display the graph classification results.

   - It plots the prediction correctness on sample graphs for both the GCN and GIN models.

   **5. Ensemble Evaluation:**

   - The EnsembleEvaluator class evaluates the performance of a combined model that averages the predictions of both GCN and GIN.

   - This provides a final accuracy comparison between individual models and the ensemble.

## How to Run

   1. Install the necessary dependencies using the following command

```bash
   pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
```

   2. Use either the Python script (chapter9.py) or the Jupyter notebook (chapter9.ipynb) to run the full graph classification pipeline.

   3. During training, the accuracy and loss metrics are printed at intervals for both GCN and GIN models. Once training is complete, visualizations and ensemble evaluations are performed.