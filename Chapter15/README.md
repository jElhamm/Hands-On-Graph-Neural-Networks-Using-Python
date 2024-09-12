# Chapter 15 - Temporal Graph Neural Networks for Traffic Speed Prediction

   This chapter implements a **Temporal Graph Neural Network (GNN)** model to predict traffic speeds using spatio-temporal data. The implementation focuses on using the `A3TGCN` layer to process time-series data from multiple traffic sensors. The chapter includes three main components: data preparation, model definition, and training & evaluation.

## Dataset

   We use the PeMSD7 traffic dataset to predict traffic speeds. The dataset contains traffic speed measurements from various sensor stations across highways. Two CSV files are used in the implementation:

   - `PeMSD7_V_228.csv`: Contains traffic speed data recorded at 5-minute intervals for 228 sensors.
   
   - `PeMSD7_W_228.csv`: Contains the distance between sensor stations.

   The adjacency matrix is computed based on the distance data, and it is used to represent the spatial connections between the sensors.

   **Download the Dataset**

   The dataset files can be downloaded from the following GitHub repository:

   - [PeMSD7 Dataset on GitHub](https://github.com/VeritasYin/STGCN_IJCAI-18)

## Files

   The code for this chapter is provided in two formats:

   - [chapter15.py](Chapter15/chapter15.py): Python script format.

   - [chapter15.ipynb](Chapter15/chapter15.ipynb): Jupyter notebook format.

## Code Overview

   **1. Data Preparation**

   The `DataPreparation` class is responsible for loading the traffic speed and distance data from CSV files. It also computes the adjacency matrix based on sensor distances and provides visualizations of the data:

   - Traffic Speed Plot: Visualizes the traffic speed time series.

   - Mean and Standard Deviation Plot: Shows the mean and standard deviation of the speed data over time.
   
   - Correlation and Distance Matrices: Visualizes the correlation between sensor speeds and the distance matrix.

   **2. Temporal GNN Model**

   The `TemporalGNN` class defines a temporal graph neural network using the `A3TGCN` layer. This model processes the time series data from multiple sensors and predicts future traffic speeds.

   **3. Model Training and Evaluation**

   The `ModelTraining` class handles the training and evaluation of the model. The training process involves minimizing the mean squared error (MSE) loss, and the model performance is evaluated using several metrics:

   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Percentage Error (MAPE)

   After training, the model predicts future traffic speeds based on historical data.

## Installation

   To run the code, ensure you have the following dependencies installed:

```bash
   !pip install -q torch-scatter~=2.1.0 torch-sparse~=0.6.16 torch-cluster~=1.6.0 torch-spline-conv~=1.2.1 torch-geometric==2.2.0 -f https://data.pyg.org/whl/torch-{torch.__version__}.html
   !pip install -q torch-geometric-temporal==0.54.0
```

## Running the Code

   1. Load the traffic speed and distance data.
   2. Visualize the data using the provided plotting methods.
   3. Compute the adjacency matrix.
   4. Train the Temporal GNN model on the data.
   5. Evaluate the model's performance using the provided metrics.

## Results

   * After training the model, you can evaluate its predictions for traffic speeds using the test dataset.
   
   * The evaluation results include MAE, RMSE, and MAPE, which measure how well the model predicts traffic speeds.


## References

```bash
   The dataset used in this chapter is from the following paper:

   Y. Li, R. Yu, C. Shahabi, and Y. Liu. "Diffusion   Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting." IJCAI 2018.
```