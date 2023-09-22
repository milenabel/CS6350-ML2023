import numpy as np
from collections import Counter
import pandas as pd

# Dataset folder storing required data files
dataset_folder = "car"  

# Construct the file paths
train_data_path = f"DecisionTree/data/{dataset_folder}/train.csv"
test_data_path = f"DecisionTree/data/{dataset_folder}/test.csv"

# Load training and test datasets without header
train_data = pd.read_csv(train_data_path, header=None)  # Updated separator to ';' and added header=None
test_data = pd.read_csv(test_data_path, header=None)    # Updated separator to ';' and added header=None

# Define attributes and label
attributes = train_data.columns[:-1].tolist()
label = train_data.columns[-1]

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

# Create lists to store errors for each combination of heuristic and depth
train_errors = []
test_errors = []

for heuristic in heuristics:
    for max_depth in range(1, 7):
        # Train decision tree using ID3 algorithm
        decision_tree = id3_algorithm(train_data, attributes, label, max_depth, heuristic)
        
        # Evaluate on training data
        train_predictions = [predict(decision_tree, instance) for _, instance in train_data.iterrows()]
        train_error = sum(train_predictions != train_data[label]) / len(train_data)
        
        # Evaluate on test data
        test_predictions = [predict(decision_tree, instance) for _, instance in test_data.iterrows()]
        test_error = sum(test_predictions != test_data[label]) / len(test_data)
        
        # Append errors to the respective lists
        train_errors.append((max_depth, heuristic, train_error))
        test_errors.append((max_depth, heuristic, test_error))

# Now, you can calculate and print average errors for each heuristic and depth combination
for heuristic in heuristics:
    print(f"Heuristic: {heuristic}")
    for max_depth in range(1, 7):
        # Calculate average train and test errors
        avg_train_error = sum(train_error for d, h, train_error in train_errors if h == heuristic and d == max_depth) / len([train_error for d, h, train_error in train_errors if h == heuristic and d == max_depth])
        avg_test_error = sum(test_error for d, h, test_error in test_errors if h == heuristic and d == max_depth) / len([test_error for d, h, test_error in test_errors if h == heuristic and d == max_depth])

        print(f"Max Depth: {max_depth}, Avg Train Error: {avg_train_error:.4f}, Avg Test Error: {avg_test_error:.4f}")
