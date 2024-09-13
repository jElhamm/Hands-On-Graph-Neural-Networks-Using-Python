# Chapter 5

   This chapter provides an in-depth exploration of node classification using two powerful models: [Multilayer Perceptron (MLP)](https://en.wikipedia.org/wiki/Multilayer_perceptron) and Vanilla **Graph Neural Network (GNN)**.
   We will implement both models using PyTorch and PyTorch Geometric libraries, and test them on two popular graph datasets, Cora and FacebookPagePage.

## Overview

   In this chapter, we demonstrate:

   - The implementation of MLP and Vanilla GNN models.

   - Comparative evaluation of the two models on the Cora and FacebookPagePage datasets.

   - Detailed dataset analysis to understand their structure and node properties.


   The chapter walks through the following key steps:

      1. Initializing random seeds for reproducibility.

      2. Loading and analyzing datasets, specifically `Cora` and `FacebookPagePage`.

      3. Defining the MLP and Vanilla GNN models, both of which will be used for node classification.

      4. Training and testing the models to evaluate their performance on node classification tasks.

      5. Creating adjacency matrices and masks for graph data representation.

      6. Visualizing results for both MLP and Vanilla GNN on the datasets, providing a comprehensive comparison of their accuracy and loss metrics.

## Files

   - [`chapter5.py`](Chapter05/chapter5.py): Python script containing the complete code for this chapter, including MLP and Vanilla GNN implementations, dataset loading, and evaluation.

   - [`chapter5.ipynb`](Chapter05/chapter5.py): Jupyter notebook that replicates the Python script in an interactive format, allowing for dynamic code execution and result visualization.

   Both files demonstrate the same underlying logic and showcase the power of combining deep learning with graph structures to perform node classification tasks.

## Key Highlights

   - **MLP Model**: A simple, two-layer perceptron used for node classification without graph connectivity information.

   - **Vanilla GNN Model**: A basic graph neural network that incorporates node connectivity for more accurate classification.

   - **Performance Comparison**: Evaluate how the MLP and Vanilla GNN models perform on different datasets, providing insights into the strengths and weaknesses of graph-aware models.

## Installation

   Make sure to install the necessary dependencies before running the code:

```bash
   pip install -q torch-scatter~=2.1.0 torch-sparse~=0.6.16 torch-cluster~=1.6.0 torch-spline-conv~=1.2.1 torch-geometric==2.2.0 -f https://data.pyg.org/whl/torch-{torch.__version__}.html
```


---

   - We hope you enjoy diving into the world of Graph Neural Networks with this chapter, gaining practical insights into node classification through hands-on code examples.