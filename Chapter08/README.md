# Chapter 8

   Chapter 8 of the book "Hands-On Graph Neural Networks Using Python" demonstrates a complete workflow for implementing and evaluating a `GraphSAGE Model` using PyTorch Geometric on the `PPI (Protein-Protein Interaction) Dataset`. The code showcases how to load data, train the model, and evaluate its performance.

## Code Structure

   **1. Setting Random Seeds**

   - The `RandomSeedSetup` class is responsible for initializing random seeds to ensure reproducibility.
   - This step is crucial for obtaining consistent results across different runs.

   **2. Graph Dataset Handling**

   - The `GraphDataset` class manages loading and printing information about the graph dataset.
   - It includes functionality to create a data loader that samples neighbors, which is essential for managing large graphs efficiently.

   **3. Graph Visualization**

   - The `GraphVisualizer` class is used to visualize subgraphs from the training data.
   - This helps in understanding the structure and characteristics of the data being used.

   **4. GraphSAGE Model**

   - The `GraphSAGEModel` class defines a GraphSAGE model using SAGEConv layers. It includes methods for training (`fit`) and testing (`test`) the model, as well as calculating accuracy.
   - This model is key to learning node representations from the graph.

   **5. PPI Data Loading**

   - The `PPIDataLoader` class handles the loading and preparation of the PPI dataset.
   - It sets up training, validation, and test data loaders tailored for the PPI dataset.

   **6. Model Training and Evaluation**

   - The `GraphSAGETrainer` class is used to initialize, train, and evaluate the GraphSAGE model.
   - It includes methods for setting up the model, training it over multiple epochs, and evaluating performance using the F1-score.

## Install

   Ensure you have the required dependencies by running the following command:

```bash
   pip install -q torch-scatter~=2.1.0 torch-sparse~=0.6.16 torch-cluster~=1.6.0 torch-spline-conv~=1.2.1 torch-geometric==2.2.0 -f https://data.pyg.org/whl/torch-{torch.__version__}.html
```

## Results

   The final output includes:

   - Visualizations of subgraphs

   - Training progress with loss and accuracy metrics

   - Evaluation results with F1-scores for test data

## Files

   - [chapter8.py](Chapter08/chapter8.py): Python script implementing the workflow.

   - [chapter8.ipynb](Chapter08/chapter8.ipynb): Jupyter Notebook version of the code.


   Feel free to explore and modify the code as needed to fit your research or application needs.