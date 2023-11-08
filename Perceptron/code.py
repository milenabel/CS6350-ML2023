import numpy as np
import pandas as pd
from collections import defaultdict

# Function to make predictions using the perceptron model
def predict(weights, x):
    activation = np.dot(weights, x)
    return 1 if activation >= 0 else -1

# Function to calculate the accuracy of the perceptron model
def calculate_accuracy(weights, data):
    correct_predictions = 0
    for i in range(len(data)):
        x = np.insert(data[i][:-1], 0, 1)  # Add bias term
        if predict(weights, x) == data[i][-1]:
            correct_predictions += 1
    return correct_predictions / len(data)

# Standard Perceptron Learning Algorithm
def standard_perceptron(training_data, epochs):
    weights = np.zeros(len(training_data[0]))
    for epoch in range(epochs):
        for i in range(len(training_data)):
            x = np.insert(training_data[i][:-1], 0, 1)  # Add bias term
            y = training_data[i][-1]
            if y * predict(weights, x) <= 0:
                weights += y * x
    return weights

# Voted Perceptron Learning Algorithm
def voted_perceptron(training_data, epochs):
    weights = np.zeros(len(training_data[0]))
    weight_list = []
    c = 0
    for epoch in range(epochs):
        for i in range(len(training_data)):
            x = np.insert(training_data[i][:-1], 0, 1)  # Add bias term
            y = training_data[i][-1]
            if y * predict(weights, x) <= 0:
                if c > 0:
                    weight_list.append((c, np.copy(weights)))
                weights += y * x
                c = 1
            else:
                c += 1
    if c > 0:
        weight_list.append((c, weights))  # Append the last one if it made any correct predictions
    return weight_list

# Average Perceptron Learning Algorithm
def average_perceptron(training_data, epochs):
    weights = np.zeros(len(training_data[0]))
    cumulative_weights = np.zeros(len(training_data[0]))
    for epoch in range(epochs):
        for i in range(len(training_data)):
            x = np.insert(training_data[i][:-1], 0, 1)  # Add bias term
            y = training_data[i][-1]
            if y * predict(weights, x) <= 0:
                weights += y * x
            cumulative_weights += weights
    return cumulative_weights / (epochs * len(training_data))

# Function to predict using the voted perceptron model
def predict_voted(weight_list, x):
    votes = 0
    for count, weights in weight_list:
        votes += count * predict(weights, x)
    return 1 if votes >= 0 else -1

# Function to evaluate the voted perceptron model
def evaluate_voted(weight_list, test_data):
    correct_predictions = 0
    for i in range(len(test_data)):
        x = np.insert(test_data[i][:-1], 0, 1)  # Add bias term
        if predict_voted(weight_list, x) == test_data[i][-1]:
            correct_predictions += 1
    return correct_predictions / len(test_data)

# Load training and test datasets without header
train_data = pd.read_csv("Perceptron/data/train.csv", header=None).values
test_data = pd.read_csv("Perceptron/data/test.csv", header=None).values

# Adjust labels to be -1 for negative class and 1 for positive class
train_data[:, -1] = np.where(train_data[:, -1] == 0, -1, 1)
test_data[:, -1] = np.where(test_data[:, -1] == 0, -1, 1)

# (a) Standard Perceptron
T = 10
standard_weights = standard_perceptron(train_data, T)
standard_test_error = 1 - calculate_accuracy(standard_weights, test_data)

# Print out the results
print("Answer for part a.")
print("Standard Perceptron learned weight vector:", standard_weights)
print("Standard Perceptron average prediction error on the test dataset:", standard_test_error)

# (b) Voted Perceptron
voted_weight_list = voted_perceptron(train_data, T)
voted_test_error = 1 - evaluate_voted(voted_weight_list, test_data)

# Sum the counts for distinct weight vectors
weight_counts = defaultdict(int)
for count, weight in voted_weight_list:
    weight_tuple = tuple(weight)
    weight_counts[weight_tuple] += count

# Create a list of tuples from the dictionary
distinct_weight_counts = list(weight_counts.items())

# Convert this list to a DataFrame
distinct_weights_df = pd.DataFrame({
    'Weights': [list(weights) for weights, _ in distinct_weight_counts],
    'Counts': [count for _, count in distinct_weight_counts]
})

# Assume 'distinct_weights_df' is your DataFrame
latex_table = distinct_weights_df.to_latex(index=False, header=True, longtable=True)

# Save the LaTeX table code to a file
with open('Perceptron/figs/vpwt.tex', 'w') as f:
    f.write(latex_table)


# Print out the results
print("Answer for part b.")
print("Distinct Voted Perceptron weight vectors and summed counts:")
print(distinct_weights_df)
distinct_weights_df.to_csv("Perceptron/figs/distinct_voted_perceptron_weights_counts.csv", index=False)
print("Voted Perceptron average test error:", voted_test_error)

# (c) Average Perceptron
average_weights = average_perceptron(train_data, T)
average_test_error = 1 - calculate_accuracy(average_weights, test_data)

# Print out the results
print("Answer for part c.")
print("Average Perceptron learned weight vector:", average_weights)
print("Average Perceptron average prediction error on the test dataset:", average_test_error)

# (d) Compare the average prediction errors
print("Answer for part d.")
print("Comparison of average prediction errors on the test dataset:")
print(f"Standard Perceptron: {standard_test_error}")
print(f"Voted Perceptron: {voted_test_error}")
print(f"Average Perceptron: {average_test_error}")
