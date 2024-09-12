# Chapter 13

   This chapter explores `Temporal Graph Neural Networks (GNNs)` by implementing various models using the PyTorch Geometric Temporal library. The chapter demonstrates how to apply these models to two real-world datasets: `WikiMaths` and `EnglandCovid`, `highlighting` key techniques such as time-series prediction and node regression.

## Introduction

   In this chapter, we implement and evaluate temporal graph neural networks (GNNs) on time-series datasets. We explore how to leverage temporal graph structures to predict outcomes at each node in dynamic graphs. The focus is on using GNNs for time-series forecasting, employing the EvolveGCNH, EvolveGCNO, and MPNNLSTM models.

## Datasets

   **WikiMaths Dataset**


   The WikiMaths Dataset consists of page visit counts from the WikiMaths page. The time-series data is split into a 50/50 ratio for training and testing. This dataset is used to evaluate the ability of temporal GNNs to predict the number of visits over time.

   - Visualization: The script visualizes the mean and standard deviation of visits, highlighting the train-test split.

   **EnglandCovid Dataset**

   The EnglandCovid Dataset contains time-series data of COVID-19 cases in England. This dataset includes a 14-day lag and is split into an 80/20 ratio for training and testing.

   - Visualization: The mean and standard deviation of COVID-19 cases are visualized, along with the train-test split.

## Implemented Models

   **EvolveGCNH**

   This model uses EvolveGCNH layers, which apply evolving GCN layers to model time-series graphs. It is specifically used for the WikiMaths dataset.

   **EvolveGCNO**

   EvolveGCNO is a variant of evolving graph convolutions that updates the GCN architecture with EvolveGCNO layers. This model is used for the EnglandCovid dataset.

   **MPNNLSTM**

   This model combines MPNN and LSTM layers for a more sophisticated treatment of temporal data in graphs, capturing long-term dependencies in the EnglandCovid dataset.

## Training and Evaluation

   **Training**

   We train each model using a custom Trainer class, which includes functionality for:

   - Optimizing with Adam.
   - Loss calculation using Mean Squared Error (MSE).
   - Backpropagation and weight updates.

## Evaluation

   Models are evaluated on the test set using MSE. We also provide visualizations of the predictions over time and regression plots to analyze the performance.

## Results

   **Mean Squared Error (MSE)**

   EnglandCovid datasets are evaluated using MSE. The error metrics for both datasets are displayed to highlight model performance.

   **Predictions**

   - WikiMaths: The script plots the model's predictions alongside actual data, with a rolling mean for smoothing.

   - EnglandCovid: The predictions are plotted over time, with standard deviations visualized to assess prediction uncertainty.

## Regression Analysis

   For both datasets, we generate regression plots comparing the predicted values against the true values, providing a visual measure of accuracy.

## Files

   The code for this chapter is provided in two formats:

   - [chapter13.py](Chapter13/chapter13.py): Python script format.

   - [chapter13.ipynb](Chapter13/chapter13.ipynb): Jupyter notebook format.

   Both files contain the same implementation of the models, training pipeline, and visualization code.