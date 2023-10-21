## Overview
'LR.py' is a Python script that performs linear regression on a given dataset using two different methods - Batch Gradient Descent (GDA) and Stochastic Gradient Descent (SGD).

## Prerequisites
- Python 3
- NumPy
- pandas
- Matplotlib

## Running the Script
Run the script with the following command:
./run.sh

## Parameters
learning_rate: The learning rate for the gradient descent algorithms. Default is 0.01.
iterations: The number of iterations for the gradient descent algorithms. Default is 150000 for GDA and 750 for SGD.
tolerance: The tolerance for convergence in the gradient descent algorithms. Default is 1e-6.

## Outputs
Best learning rate for both GDA and SGD.
Learned weight vectors for both GDA and SGD.
Plots of the cost function value against the number of iterations for both GDA and SGD, saved as gda.png and sgd.png respectively (seen figs folder).
Test cost for both GDA and SGD.
Optimal weight vector calculated analytically.