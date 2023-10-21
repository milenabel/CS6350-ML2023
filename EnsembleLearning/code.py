import numpy as np
from collections import Counter
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# Dataset folder storing required data files
dataset_folder = "bank"  # Update to the "bank" dataset folder

# Construct the file paths
train_data_path = f"EnsembleLearning/data/train.csv"
test_data_path = f"EnsembleLearning/data/test.csv"


# Load training and test datasets without header
train_data = pd.read_csv(train_data_path, header=None, sep=",")
test_data = pd.read_csv(test_data_path, header=None, sep=",")

# Define attributes and label
attributes = train_data.columns[:-1].tolist()
label = train_data.columns[-1]

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

def decision_stump(data, attributes, label, weights):
    best_attribute, best_threshold, max_info_gain, best_left_class, best_right_class = None, None, float('-inf'), None, None
    total_weight = np.sum(weights)
    
    base_entropy = entropy(data[label], weights)
    
    for attribute in attributes:
        subsets = []
        if data[attribute].dtype == 'bool':
            subsets = [(data[data[attribute] == True], data[data[attribute] == False])]
        elif data[attribute].dtype == 'object':  # if the attribute is a string
            unique_values = data[attribute].unique()
            for value in unique_values:
                left_subset = data[data[attribute] == value]
                right_subset = data[data[attribute] != value]
                subsets.append((left_subset, right_subset))
        else:
            unique_values = sorted(data[attribute].unique())
            thresholds = [(unique_values[i] + unique_values[i + 1]) / 2 for i in range(len(unique_values) - 1)]
            
            for threshold in thresholds:
                left_subset = data[data[attribute] <= threshold]
                right_subset = data[data[attribute] > threshold]
                subsets.append((left_subset, right_subset))
        
        for subset1, subset2 in subsets:
            left_class = majority_class(subset1[label], weights[subset1.index])
            right_class = majority_class(subset2[label], weights[subset2.index])
            
            subset1_entropy = entropy(subset1[label], weights[subset1.index])
            subset2_entropy = entropy(subset2[label], weights[subset2.index])
            
            weight1 = np.sum(weights[subset1.index]) / total_weight
            weight2 = np.sum(weights[subset2.index]) / total_weight
            
            info_gain = base_entropy - (weight1 * subset1_entropy + weight2 * subset2_entropy)
            
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_attribute = attribute
                best_left_class = left_class
                best_right_class = right_class
                best_subset1 = subset1
                best_subset2 = subset2
    
    return best_attribute, best_left_class, best_right_class, max_info_gain, best_subset1, best_subset2

def entropy(labels, weights):
    if weights is None:
        weights = np.ones(len(labels))
    total_weight = np.sum(weights)
    label_counts = labels.value_counts(normalize=True)
    entropy = -np.sum(label_counts * np.log2(label_counts)) * (total_weight / len(labels))
    return entropy

def majority_class(labels, weights):
    weighted_counts = {}
    unique_labels = labels.unique()
    for label in unique_labels:
        weighted_counts[label] = np.sum(weights[labels == label])
    return max(weighted_counts, key=weighted_counts.get)

def adaboost(train_data, test_data, attributes, label, T):
    weights = np.ones(len(train_data)) / len(train_data)
    classifiers = []
    alpha_values = []
    stump_train_errors = []  
    stump_test_errors = []  
    
    for t in range(T):
        stump, left_class, right_class, max_info_gain, left_subset, right_subset = decision_stump(train_data, attributes, label, weights)
#         error = 1 - accuracy_score(left_subset[label], [left_class] * len(left_subset)) * np.sum(weights[left_subset.index]) / np.sum(weights)
        train_error = 1 - (accuracy_score(left_subset[label], [left_class] * len(left_subset)) * np.sum(weights[left_subset.index]) + 
                 accuracy_score(right_subset[label], [right_class] * len(right_subset)) * np.sum(weights[right_subset.index])) / np.sum(weights)
        test_error = 1 - accuracy_score(test_data[label], [left_class if row[stump] else right_class for index, row in test_data.iterrows()])
        alpha = 0.5 * np.log((1 - train_error) / max(train_error, 1e-10))
        classifiers.append(((left_subset, right_subset), left_class, right_class, alpha))  # Store stump as tuple
        alpha_values.append(alpha)
        
        weights[left_subset[left_subset[label] == left_class].index] *= np.exp(-alpha)
        weights[right_subset[right_subset[label] != right_class].index] *= np.exp(alpha)
        weights /= np.sum(weights)
        
#         stump_errors.append(error)  # Add this line
        stump_train_errors.append(train_error)  # Existing line
        stump_test_errors.append(test_error)  # Add this line for stump test errors

    train_errors = []
    test_errors = []
    
    for t in range(1, T + 1):
        train_predictions = predict_adaboost(classifiers[:t], train_data)
        test_predictions = predict_adaboost(classifiers[:t], test_data)
        
        train_error = 1 - accuracy_score(train_data[label], train_predictions)
        test_error = 1 - accuracy_score(test_data[label], test_predictions)
        
        train_errors.append(train_error)
        test_errors.append(test_error)
#         print("Iteration:", t)
#         print("Weights before updating:", weights)
#         print("Train Error:", train_error)
#         print("Alpha:", alpha)
#         print("Weights after updating:", weights)

        
#     return train_errors, test_errors, stump_errors  # Add stump_errors
    return train_errors, test_errors, stump_train_errors, stump_test_errors  

def predict_adaboost(classifiers, data):
    predictions = np.zeros(len(data), dtype=float)

    # Find all unique classes across all classifiers
    unique_classes = np.unique([cls for _, left_class, right_class, _ in classifiers for cls in [left_class, right_class]])
    class_to_numeric = {'no': -1, 'yes': 1}
    numeric_to_class = {-1: 'no', 1: 'yes'}

    for (stump, left_class, right_class, alpha) in classifiers:
        subset1, subset2 = stump
        numeric_predictions1 = np.full(len(data), class_to_numeric[left_class])
        numeric_predictions2 = np.full(len(data), class_to_numeric[right_class])
        predictions += alpha * np.where(data[subset1.columns[0]].isin(subset1[subset1.columns[0]]), numeric_predictions1, numeric_predictions2)

#     print("Numeric to class mapping:", numeric_to_class)
#     print("Predictions:", predictions)
# #     print("Sign of predictions:", [np.sign(pred) for pred in predictions])

    return np.array([numeric_to_class[int(np.sign(pred))] for pred in predictions])

# Vary the number of iterations T from 1 to 500
# T_values = range(1, 501)

T_values = range(1, 501)
train_errors = []
test_errors = []
stump_train_errors = []
stump_test_errors = []

for T in T_values:
#     print('iteration')
#     print("Unique classes in the data:", np.unique(train_data[label]))
    train_error, test_error, stump_train_error, stump_test_error = adaboost(train_data, test_data, attributes, label, T)  
    train_errors.append(train_error[-1])
    test_errors.append(test_error[-1])
    stump_train_errors.append(stump_train_error[-1])
    stump_test_errors.append(stump_test_error[-1])

# First figure: training and test errors vs. T
plt.figure(figsize=(10, 6))
plt.plot(T_values, train_errors, label='Training Error')
plt.plot(T_values, test_errors, label='Test Error')
plt.xlabel('Number of Iterations (T)')
plt.ylabel('Error')
plt.title('Training and Test Errors vs. Number of Iterations')
plt.legend()
plt.grid(True)
plt.show()

def bootstrap_sample(data, random_state=None):
    sample = data.sample(n=len(data), replace=True, random_state=random_state)
    return sample

def train_decision_tree(data, attributes, label):
    """Train a decision tree using the given data."""
    weights = np.ones(len(data))  # Create a uniform weight array
    attribute, left_class, right_class, _, _, _ = decision_stump(data, attributes, label, weights)
    return attribute, left_class, right_class


def predict_decision_tree(tree, data):
    """Predict using a decision tree."""
    attribute, left_class, right_class = tree
    return [left_class if row[attribute] else right_class for _, row in data.iterrows()]

def bagged_trees(train_data, test_data, attributes, label, num_trees):
    """Train and predict using bagged trees."""
    trees = []
    
    # 1. Bootstrap sampling and training trees
    for i in range(num_trees):
        sample = bootstrap_sample(train_data, random_state=i)
        tree = train_decision_tree(sample, attributes, label)
        trees.append(tree)

    # 2. Predictions
    train_predictions = []
    for tree in trees:
        train_predictions.append(predict_decision_tree(tree, train_data))

    test_predictions = []
    for tree in trees:
        test_predictions.append(predict_decision_tree(tree, test_data))

    # Majority vote for predictions
    final_train_predictions = [Counter([train_predictions[j][i] for j in range(num_trees)]).most_common(1)[0][0] for i in range(len(train_data))]
    final_test_predictions = [Counter([test_predictions[j][i] for j in range(num_trees)]).most_common(1)[0][0] for i in range(len(test_data))]

    return final_train_predictions, final_test_predictions

# Vary the number of trees from 1 to 500
num_trees_values = range(1, 501)
bagged_train_errors = []
bagged_test_errors = []

for num_trees in num_trees_values:
    print("iteration")
    train_preds, test_preds = bagged_trees(train_data, test_data, attributes, label, num_trees)
    bagged_train_errors.append(1 - accuracy_score(train_data[label], train_preds))
    bagged_test_errors.append(1 - accuracy_score(test_data[label], test_preds))

# Plotting Bagged Trees training and test errors
plt.figure(figsize=(10, 6))
plt.plot(num_trees_values, bagged_train_errors, label='Bagged Trees Training Error')
plt.plot(num_trees_values, bagged_test_errors, label='Bagged Trees Test Error')
plt.xlabel('Number of Trees')
plt.ylabel('Error')
plt.title('Bagged Trees Training and Test Errors vs. Number of Trees')
plt.legend()
plt.grid(True)
plt.show()
