from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# Loading the training and test datasets
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

def sigmoid_stable(z):
    # Clipping input to avoid overflow in exponential
    z = np.clip(z, -50, 50)
    return 1 / (1 + np.exp(-z))

def logistic_regression_SGD_modified(X, y, learning_rate, d, T, variance, map_estimation=True):
    # Initialize weights and bias
    w = np.zeros(X.shape[1])
    b = 0

    # List to store the objective function values
    objective_values = []

    for epoch in range(T):
        # Shuffle the dataset
        shuffled_indices = np.random.permutation(X.shape[0])
        X_shuffled = X[shuffled_indices]
        y_shuffled = y[shuffled_indices]

        for i in range(X.shape[0]):
            xi = X_shuffled[i]
            yi = y_shuffled[i]

            # Update learning rate
            gamma_t = learning_rate / (1 + (learning_rate / d) * epoch)

            # Compute prediction
            pred = sigmoid_stable(np.dot(xi, w) + b)

            # Update rule for weights and bias
            if map_estimation:
                # MAP estimation with Gaussian prior
                w_gradient = -(yi - pred) * xi + (w / variance)
            else:
                # ML estimation
                w_gradient = -(yi - pred) * xi

            b_gradient = -(yi - pred)

            w -= gamma_t * w_gradient
            b -= gamma_t * b_gradient

        # Compute the objective function value after each epoch
        preds = sigmoid_stable(np.dot(X, w) + b)
        preds = np.clip(preds, 1e-10, 1-1e-10)  # Avoid log(0)
        if map_estimation:
            objective = -np.sum(y * np.log(preds) + (1 - y) * np.log(1 - preds)) + (1 / (2 * variance)) * np.sum(w**2)
        else:
            objective = -np.sum(y * np.log(preds) + (1 - y) * np.log(1 - preds))

        objective_values.append(objective)

    return w, b, objective_values

# Data preparation
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Parameters for SGD
learning_rate = 0.0005  # Initial learning rate, to be tuned
d = 0.003  # Parameter for learning rate schedule, to be tuned
T = 100  # Maximum number of epochs
variances = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]  # Variance values for MAP estimation


# Running the modified MAP estimation for different variances
map_results_modified = {}
for variance in variances:
    w, b, objective_values = logistic_regression_SGD_modified(X_train, y_train, learning_rate, d, T, variance, map_estimation=True)
    
    # Predictions for training and test data
    train_preds = sigmoid_stable(np.dot(X_train, w) + b) >= 0.5
    test_preds = sigmoid_stable(np.dot(X_test, w) + b) >= 0.5

    # Calculating training and test errors
    train_error = 1 - accuracy_score(y_train, train_preds)
    test_error = 1 - accuracy_score(y_test, test_preds)

    map_results_modified[variance] = {
        "train_error": train_error,
        "test_error": test_error,
#         "objective_curve": objective_values
    }

# map_results_modified.keys()  # Checking the keys of the results dictionary to confirm the process completion
map_results_modified

# Implementing the maximum likelihood (ML) estimation

ml_results = {}
for variance in variances:
    w, b, objective_values = logistic_regression_SGD_modified(X_train, y_train, learning_rate, d, T, variance, map_estimation=False)
    
    # Predictions for training and test data
    train_preds = sigmoid_stable(np.dot(X_train, w) + b) >= 0.5
    test_preds = sigmoid_stable(np.dot(X_test, w) + b) >= 0.5

    # Calculating training and test errors
    train_error = 1 - accuracy_score(y_train, train_preds)
    test_error = 1 - accuracy_score(y_test, test_preds)

    ml_results[variance] = {
        "train_error": train_error,
        "test_error": test_error,
#         "objective_curve": objective_values
    }

# Plotting the results for ML estimation
# fig, axs = plt.subplots(len(variances), 1, figsize=(10, 20))

for i, variance in enumerate(variances):
    results = ml_results[variance]
    train_error = results["train_error"]
    test_error = results["test_error"]
#     objective_curve = results["objective_curve"]

#     axs[i].plot(objective_curve, label=f'Variance: {variance}')
#     axs[i].set_title(f'ML Estimation Objective Function Curve for Variance {variance}\nTrain Error: {train_error:.4f}, Test Error: {test_error:.4f}')
#     axs[i].set_xlabel('Epochs')
#     axs[i].set_ylabel('Objective Function Value')
#     axs[i].legend()

# plt.tight_layout()
# plt.show()

ml_results