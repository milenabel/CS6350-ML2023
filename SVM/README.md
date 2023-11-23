## code1.py

## Overview
'code1.py' is a Python script implementing a Support Vector Machine (SVM) using stochastic sub-gradient descent with an adaptive learning rate. The script is designed to optimize SVM parameters for binary classification tasks. It adjusts learning rate parameters dynamically to achieve convergence, evaluates different regularization strengths, and provides comprehensive outputs including model parameters and error metrics.

## Prerequisites
- Python 3
- NumPy
- pandas
- scikit-learn (for data shuffling)

## Running the Script
Run the script with the following command:
./run.sh

## Parameters
- 'C': Regularization parameter. The script tests multiple values to understand their impact on model performance.
- 'T': The number of training epochs.
- 'initial_gamma_0' and 'initial_a': Starting values for the learning rate and its decay parameter, respectively.

## Inputs
The script processes two datasets:
- Training dataset (train.csv): For training the SVM model.
- Test dataset (test.csv): For evaluating model performance.

## Outputs
The script outputs the following:
- Learned model parameters (weights) for different values of C.
- Training and testing errors for different regularization strengths.
- Convergence status and the adaptive learning rate parameters (gamma_0, a) for each model.
- The differences in parameters and error metrics between two different learning rate schedules.

## Functionality
- Adaptive Learning Rate: Adjusts gamma_0 and a to ensure convergence of the model.
- Model Training and Evaluation: Trains the SVM model on the training set and evaluates its performance on the test set.
- Error and Parameter Analysis: Provides a detailed analysis of training/testing errors and model parameter differences under varying conditions.


## code2.py

## Overview
'code2.py' is a Python script for implementing SVM in the dual domain. It uses the scipy.optimize.minimize function to solve the dual problem of SVM with linear and Gaussian kernels. The script experiments with different values of regularization parameter C and kernel parameter gamma, and it calculates training and test errors for each combination.

## Prerequisites
- Python 3
- NumPy
- pandas
- SciPy (for optimization)

## Running the Script
Run the script with the following command:
./run.sh

## Parameters
- 'C': Regularization parameter, with multiple values tested.
- 'gamma': Kernel parameter for the Gaussian kernel, with multiple values tested.

## Inputs
The script processes two datasets:
- Training dataset (train.csv): For training the SVM model.
- Test dataset (test.csv): For evaluating model performance.

## Outputs
The script outputs the following:
- Optimized alpha values from the dual problem.
- Model parameters (weights w and bias b) for different C and gamma values.
- Training and test errors for each combination of C and gamma.
- Analysis of support vectors and their overlaps for different gamma values at a specific C.

## Functionality
- Kernel SVM Implementation: Implements both linear and Gaussian kernel SVM in the dual form.
- Parameter Optimization: Uses constrained optimization to find the best alpha values.
- Error Analysis: Calculates and outputs the training and testing errors.
- Support Vector Analysis: Identifies support vectors and analyzes their overlaps for different kernel parameters, providing insights into the model's complexity and generalization capability.