## code.py

## Overview
'code.py; is a Python script for implementing a simple feedforward Neural Network (NN) with three layers: input, hidden, and output. The script focuses on forward and backward propagation, including weight updates using stochastic gradient descent (SGD). It demonstrates the fundamentals of a neural network, including activation functions, loss calculations, and weight adjustments for learning.

## Prerequisites
- Python 3
- NumPy
- pandas (for data loading)

## Running the Script
Run the script with the following command:
./run.sh

## Inputs
Training and test datasets (train.csv and test.csv) are loaded to demonstrate the network's training and evaluation.

## Functionality
- Single Example Training: Demonstrates forward and backward passes for a single data example.
- Batch Training: Shows training over multiple epochs with the entire dataset using SGD.
- Error Calculation: Computes the mean squared error for predictions.
- Parameter Tuning: Allows tuning of hyperparameters like hidden layer size, learning rate, and epochs.

## Outputs
The script primarily outputs the training process details, including forward pass outputs, gradients during backpropagation, and the error after training.

## bonus2.py

## Overview
'bonus2.py' is a Python script that implements logistic regression with stochastic gradient descent (SGD) and Maximum A Posteriori (MAP) estimation. The script includes modifications for MAP estimation with Gaussian prior and maximum likelihood (ML) estimation. It is designed to explore the impact of different variance values on logistic regression's performance.

## Prerequisites
- Python 3
- NumPy
- pandas (for data loading)
- scikit-learn (for accuracy calculation)
 
## Running the Script
Run the script with the following command:
./run.sh

## Inputs
Training and test datasets (train.csv and test.csv) for logistic regression.

Functionality
Training and Testing: Trains the logistic regression model using the training set and evaluates it on the test set.
Error Calculation: Computes training and test errors for model evaluation.
Hyperparameter Tuning: Includes tuning of learning rate, decay parameter, and variance values.
Outputs
Outputs include training and test errors for different settings of variance in MAP estimation and for the ML estimation.
Provides insights into the influence of variance on the logistic regression model's performance.
Usage
The script is particularly useful for understanding the differences in logistic regression's performance under MAP estimation with various variance settings compared to standard ML estimation.





