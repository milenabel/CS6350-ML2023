## Overview
'code.py' is a Python script that implements three variations of the Perceptron algorithm: Standard Perceptron, Voted Perceptron, and Average Perceptron. The script performs data preprocessing, executes each of the algorithms on training data, evaluates them on test data, and stores the results, including a distinct count of weight vectors learned by the Voted Perceptron, both as a CSV file and a LaTeX table.

## Prerequisites
- Python 3
- NumPy
- pandas
- Jinja2

## Running the Script
Run the script with the following command:
./run.sh

## Parameters
'T': The number of training epochs. It is set to 10 in the current script but can be modified according to the desired number of iterations for the learning algorithms.

## Inputs
The script reads two datasets:
- Training dataset (train.csv): Used to train the Perceptron models.
- Test dataset (test.csv): Used to evaluate the models and calculate the prediction error.

## Outputs
The script outputs the following:
- Learned weight vector for the Standard Perceptron and its average prediction error on the test dataset.
- List of distinct weight vectors along with their respective counts for the Voted Perceptron, saved as a CSV file and as a LaTeX table in the 'Perceptron/figs' directory.
- Average prediction error for the Voted Perceptron on the test dataset.
- Learned weight vector for the Average Perceptron and its average prediction error on the test dataset.
- A comparison of the average prediction errors of all three Perceptron algorithms on the test dataset.

The output is displayed on the console, and the distinct weight vectors with counts are also saved externally in two formats for further analysis or documentation purposes.

## Functionality
The script functions in the following manner:

- Data Preprocessing: It loads the training and test data, adding a bias term to the feature set and converting the class labels to -1 and 1.
- Standard Perceptron: It runs the Standard Perceptron algorithm, printing out the learned weights and computing the test error.
- Voted Perceptron: It runs the Voted Perceptron algorithm, accumulating the weights and their counts. It then computes the test error and saves the distinct weights and their counts to a CSV file and as a LaTeX table.
- Average Perceptron: It runs the Average Perceptron algorithm, calculating and printing out the average of the weights over all epochs and the test error.
- Comparison: Finally, it compares the test errors of all three algorithms, summarizing their performance on the given dataset.

Each step involves careful handling of data and utilizes NumPy for efficient numerical computations, and pandas for data manipulation and output formatting.