## Overview
'LR.py' is a Python script that  uses AdaBoost and bagged trees algorithms to learn decision trees on a given dataset. The script performs data preprocessing, trains decision trees, and then plots errors with varying iterations and number of trees. The dataset used is specified in the dataset_folder variable within the script.

## Prerequisites
- Python 3
- NumPy
- pandas
- Matplotlib
- sklearn

## Running the Script
Run the script with the following command:
./run.sh

## Parameters
dataset_folder: The folder containing the dataset files. The default is set to "bank".
T_values: The range of iterations for AdaBoost. The default is set from 1 to 500.
num_trees_values: The range for the number of trees for bagged trees algorithm. The default is set from 1 to 500.

## Outputs
The script will output two plots:
- Training and test errors vs. the number of iterations for AdaBoost.
- Training and test errors vs. the number of trees for bagged trees.
These plots help in visualizing the performance of the algorithms with varying iterations and number of trees, and can be used to determine the optimal values for these parameters.
However, this code has potential bugs in it, so the output seems to be off.