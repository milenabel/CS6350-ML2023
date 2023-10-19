import numpy as np
from collections import Counter
import pandas as pd
from sklearn.metrics import accuracy_score

# Dataset folder storing required data files
dataset_folder = "bank"  # Update to the "bank" dataset folder

# Construct the file paths
train_data_path = f"DecisionTree/data/{dataset_folder}/train.csv"
test_data_path = f"DecisionTree/data/{dataset_folder}/test.csv"

# Load training and test datasets without header
train_data = pd.read_csv(train_data_path, header=None, sep=",")
test_data = pd.read_csv(test_data_path, header=None, sep=",")

# Define attributes and label
attributes = train_data.columns[:-1].tolist()
label = train_data.columns[-1]

# Function to handle missing values by completing with majority value
# def handle_missing_values(data, attributes):
#     for attr in attributes:
#         majority_value = data[attr].mode()[0]  # Find the majority value
#         data[attr] = data[attr].replace('unknown', majority_value)  # Replace 'unknown' with majority value
#     return data

# # Handle missing values in both training and test data
# train_data = handle_missing_values(train_data, attributes)
# test_data = handle_missing_values(test_data, attributes)

# train_data.head()
# test_data.head()

# Function to convert numerical attributes to binary based on median
def convert_numerical_to_binary(data, threshold_indices):
    for idx in threshold_indices:
        if data[idx].dtype == 'float64' or data[idx].dtype == 'int64':
            median_threshold = data[idx].median()
            data[idx] = data[idx] > median_threshold
    return data

# Identify numerical attributes by their indices
numerical_indices = [0, 5, 9, 10, 11, 12, 13, 14]

# Convert numerical attributes to binary based on the medians
train_data = convert_numerical_to_binary(train_data, numerical_indices)
test_data = convert_numerical_to_binary(test_data, numerical_indices)

# train_data.head()
# test_data.head()

# Implement ID3 Algorithm
def id3_algorithm(data, attributes, label, max_depth, heuristic):
    def majority_class(data, label):
        # Return the class with the majority count in the data
        label_counts = Counter(data[label])
        return label_counts.most_common(1)[0][0]

    def information_gain(data, attribute, label):
        # Calculate the information gain for a given attribute
        total_entropy = entropy(data[label])
        attribute_entropy = 0.0

        for value in data[attribute].unique():
            subset = data[data[attribute] == value]
            weight = len(subset) / len(data)
            attribute_entropy += weight * entropy(subset[label])

        return total_entropy - attribute_entropy

    def entropy(data):
        # Calculate the entropy of a dataset
        label_counts = Counter(data)
        total_instances = len(data)
        entropy = 0.0

        for count in label_counts.values():
            probability = count / total_instances
            entropy -= probability * np.log2(probability)

        return entropy

    def gini_index(data, attribute, label):
        # Calculate the Gini index for a given attribute
        total_gini = gini(data[label])
        attribute_gini = 0.0

        for value in data[attribute].unique():
            subset = data[data[attribute] == value]
            weight = len(subset) / len(data)
            attribute_gini += weight * gini(subset[label])

        return total_gini - attribute_gini

    def gini(data):
        # Calculate the Gini index of a dataset
        label_counts = Counter(data)
        total_instances = len(data)
        gini_index = 1.0

        for count in label_counts.values():
            probability = count / total_instances
            gini_index -= probability ** 2

        return gini_index

    def id3_recursive(data, attributes, label, max_depth, heuristic):
        # Base cases
        if len(data[label].unique()) == 1:
            return data[label].iloc[0]
        if len(attributes) == 0 or max_depth == 0:
            return majority_class(data, label)

        # Select the best attribute based on the chosen heuristic
        if heuristic == 'information_gain':
            best_attribute = max(attributes, key=lambda attr: information_gain(data, attr, label))
        elif heuristic == 'majority_error':
            best_attribute = min(attributes, key=lambda attr: majority_error(data, attr, label))
        elif heuristic == 'gini_index':
            best_attribute = max(attributes, key=lambda attr: gini_index(data, attr, label))

        tree = {best_attribute: {}}
        remaining_attributes = [attr for attr in attributes if attr != best_attribute]

        for value in data[best_attribute].unique():
            subset = data[data[best_attribute] == value]
            tree[best_attribute][value] = id3_recursive(subset, remaining_attributes, label, max_depth - 1, heuristic)

        return tree

    def majority_error(data, attribute, label):
        # Calculate the majority error for a given attribute
        total_instances = len(data)
        error = 0.0

        for value in data[attribute].unique():
            subset = data[data[attribute] == value]
            label_counts = Counter(subset[label])
            majority_class_count = max(label_counts.values())
            error += (len(subset) - majority_class_count) / total_instances

        return error

    return id3_recursive(data, attributes, label, max_depth, heuristic)

# Prediction
def predict(tree, instance):
    # Make predictions using the decision tree
    if isinstance(tree, str):
        return tree

    root_attribute = list(tree.keys())[0]
    root_value = instance[root_attribute]

    if root_value not in tree[root_attribute]:
        # If the value is not in the tree, return the majority class
        return list(tree[root_attribute].values())[0]

    # Recursively traverse the tree
    subtree = tree[root_attribute][root_value]
    return predict(subtree, instance)

# Define the list of heuristics to use
heuristics = ['information_gain', 'majority_error', 'gini_index']

for heuristic in heuristics:
    for max_depth in range(1, 17):  # Vary maximum tree depth from 1 to 16
        if max_depth > len(attributes):
            continue  # Skip if max depth exceeds the number of attributes
            
        # Train decision tree using ID3 algorithm
        decision_tree = id3_algorithm(train_data, attributes, label, max_depth, heuristic)
        
        # Evaluate on training data
        train_predictions = [predict(decision_tree, instance) for _, instance in train_data.iterrows()]
        train_error = sum(train_predictions != train_data[label]) / len(train_data)
        
        # Evaluate on test data
        test_predictions = [predict(decision_tree, instance) for _, instance in test_data.iterrows()]
        test_error = sum(test_predictions != test_data[label]) / len(test_data)
        
        # Print results for each combination of depth and heuristic
        print(f"Heuristic: {heuristic}, Max Depth: {max_depth}")
        print(f"Train Error: {train_error:.4f}")
        print(f"Test Error: {test_error:.4f}")
        print()  # Add a new line between results