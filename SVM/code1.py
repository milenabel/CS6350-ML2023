import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def svm_sgd_with_adaptive_learning_rate(X, y, C, T, learning_rate_schedule, initial_gamma_0=0.01, initial_a=0.01, convergence_threshold=1e-3):
    """
    Implement SVM using stochastic sub-gradient descent with adaptive learning rate and convergence check.

    :param X: Training data features
    :param y: Training data labels
    :param C: Regularization parameter
    :param T: Maximum number of epochs
    :param learning_rate_schedule: Function to compute the learning rate at each step
    :param initial_gamma_0: Initial learning rate
    :param initial_a: Initial parameter for learning rate schedule
    :param convergence_threshold: Threshold for convergence
    :return: Model parameters (weights), objective function values, convergence status, final gamma_0, and final a
    """
    gamma_0_values = [0.1, 0.01, 0.005]
    a_values = [1, 100, 1000, 5000]

    for gamma_0 in gamma_0_values:
        for a in a_values:
            n_samples, n_features = X.shape
            w = np.zeros(n_features)  # Initialize weights
            objective_values = []
            converged = False

            for epoch in range(T):
                X, y = shuffle(X, y)  # Shuffle the data at the start of each epoch
                for i in range(n_samples):
                    t = epoch * n_samples + i  # Total number of iterations
                    gamma_t = learning_rate_schedule(gamma_0, a, t)  # Learning rate at iteration t
                    if y[i] * np.dot(X[i], w) < 1:
                        w = w - gamma_t * (w - C * y[i] * X[i])
                    else:
                        w = w - gamma_t * w

                    # Objective function value
                    hinge_loss = np.maximum(0, 1 - y * np.dot(X, w)).mean()
                    objective_value = 0.5 * np.dot(w, w) + C * hinge_loss
                    objective_values.append(objective_value)

                    # Check for convergence
                    if i > 0 and abs(objective_values[-1] - objective_values[-2]) < convergence_threshold:
                        converged = True
                        break

                if converged:
                    break

            if converged:
                return w, objective_values, converged, gamma_0, a

    # If convergence is not achieved with any of the parameter settings
    return w, objective_values, converged, gamma_0, a

# Define the learning rate schedules
def learning_rate_schedule_1(gamma_0, a, t):
    return gamma_0 / (1 + (gamma_0 / a) * t)

def learning_rate_schedule_2(gamma_0, a, t):
    return gamma_0 / (1 + t)

# Load the data
train_data = pd.read_csv('data/train.csv', header=None)
test_data = pd.read_csv('data/test.csv', header=None)

# Convert labels to {1, -1}
train_data.iloc[:, -1] = train_data.iloc[:, -1].apply(lambda x: 1 if x == 0 else -1)
test_data.iloc[:, -1] = test_data.iloc[:, -1].apply(lambda x: 1 if x == 0 else -1)


# Extract features and labels
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Hyperparameters
Cs = [100 / 873, 500 / 873, 700 / 873]
T = 100  # Maximum number of epochs

# Placeholder for results
results = []

for C in Cs:
    # First learning rate schedule
    w1, objective_values_1, converged_1, gamma_0_1, a_1 = svm_sgd_with_adaptive_learning_rate(X_train, y_train, C, T, learning_rate_schedule_1)
    train_error_1 = np.mean(y_train != np.sign(np.dot(X_train, w1)))
    test_error_1 = np.mean(y_test != np.sign(np.dot(X_test, w1)))

    # Second learning rate schedule
    w2, objective_values_2, converged_2, gamma_0_2, a_2 = svm_sgd_with_adaptive_learning_rate(X_train, y_train, C, T, learning_rate_schedule_2)
    train_error_2 = np.mean(y_train != np.sign(np.dot(X_train, w2)))
    test_error_2 = np.mean(y_test != np.sign(np.dot(X_test, w2)))

    # Collect results
    results.append({
        'C': C,
        'Schedule 1': {'Train Error': train_error_1, 'Test Error': test_error_1, 'Converged': converged_1, 'gamma_0': gamma_0_1, 'a': a_1},
        'Schedule 2': {'Train Error': train_error_2, 'Test Error': test_error_2, 'Converged': converged_2, 'gamma_0': gamma_0_2, 'a': a_2},
        'Parameter Difference': np.linalg.norm(w1 - w2),
        'Train Error Difference': abs(train_error_1 - train_error_2),
        'Test Error Difference': abs(test_error_1 - test_error_2)
    })
    
results