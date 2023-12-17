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

## bonus1.py

## Overview
'bonus1.py' is a Python script that implements a customizable deep neural network using PyTorch. The script allows for experimentation with different network depths, widths, and activation functions. It includes data preprocessing, model training, and evaluation phases, making it a comprehensive tool for exploring the impact of neural network architecture on binary classification tasks.

## Prerequisites
- Python 3
- PyTorch
- pandas (for data loading)
- scikit-learn (for data scaling)
- torch.utils.data (for data loading and batching)

## Running the Script
Run the script with the following command:
./run.sh

## Inputs
Training and test datasets (train.csv and test.csv) for binary classification.

## Functionality
- Model Customization: Allows for varying the depth and width of the neural network, as well as choosing between different activation functions (Tanh or ReLU).
- Weight Initialization: Adopts Xavier or He initialization based on the chosen activation function.
- Error Calculation: Computes training and test errors to evaluate model performance.
- Experimentation with Architecture: Facilitates experimentation with different network architectures to study their impact on model accuracy.

## Outputs
The script outputs training and test errors for each combination of network depth, width, and activation function, providing insights into how these parameters influence model performance.

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

## Functionality
- Training and Testing: Trains the logistic regression model using the training set and evaluates it on the test set.
- Error Calculation: Computes training and test errors for model evaluation.
- Hyperparameter Tuning: Includes tuning of learning rate, decay parameter, and variance values.

## Outputs
Outputs include training and test errors for different settings of variance in MAP estimation and for the ML estimation.
Provides insights into the influence of variance on the logistic regression model's performance.





