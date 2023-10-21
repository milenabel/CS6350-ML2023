import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Construct the file paths
train_data_path = f"LinearRegression/data/train.csv"
test_data_path = f"LinearRegression/data/test.csv"

# Load training and test datasets without header
train_data = pd.read_csv(train_data_path, header=None, sep=",")
test_data = pd.read_csv(test_data_path, header=None, sep=",")

# Separate features and output
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Add a column of ones for the bias term
X_train = np.concatenate([np.ones((X_train.shape[0], 1)), X_train], axis=1)
X_test = np.concatenate([np.ones((X_test.shape[0], 1)), X_test], axis=1)

def cost_function(X, y, weights):
    m = len(y)
    cost = (1 / (2 * m)) * np.sum((X.dot(weights) - y) ** 2)
    return cost

def batch_gradient_descent(X, y, learning_rate=0.01, iterations=150000, tolerance=1e-6):
    m, n = X.shape
    weights = np.zeros(n)
    cost_history = []
    for i in range(iterations):
        gradients = (1 / m) * X.T.dot(X.dot(weights) - y)
        new_weights = weights - learning_rate * gradients
        cost = cost_function(X, y, new_weights)
        cost_history.append(cost)
        weight_diff = np.linalg.norm(new_weights - weights)
        if weight_diff < tolerance:
            print(f'Convergence achieved at iteration {i+1} with learning rate {learning_rate}')
            break
        weights = new_weights
    return weights, cost_history

# Tune the learning rate
learning_rates = [1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]
best_lr = 0
best_weights = None
best_cost = float('inf')
max_cost_history = []

for lr in learning_rates:
    weights, cost_history = batch_gradient_descent(X_train, y_train, learning_rate=lr)
    final_cost = cost_history[-1]
    if final_cost < best_cost:
        best_lr = lr
        best_weights = weights
        best_cost = final_cost
        max_cost_history = cost_history if len(cost_history) > len(max_cost_history) else max_cost_history
    print(f'Learning Rate: {lr}, Final Cost: {final_cost}, Number of Iterations: {len(cost_history)}')

# Report the learned weight vector and learning rate
print(f'Best Learning Rate: {best_lr}')
print(f'Learned GDA Weight Vector: {best_weights}')

# Plot the cost function value against the number of iterations
plt.plot(range(len(max_cost_history)), max_cost_history)
plt.xlabel('Number of Iterations')
plt.ylabel('Cost Function Value')
plt.title('Cost Function Value vs Number of Iterations')
plt.savefig('gda.png') 
plt.show()

# Evaluate the model on the test data
test_cost = cost_function(X_test, y_test, best_weights)
print(f'Test Cost: {test_cost}')

def stochastic_gradient_descent(X, y, learning_rate=0.01, iterations=750, tolerance=1e-6):
    m, n = X.shape
    weights = np.zeros(n)
    cost_history = []
    prev_cost = float('inf')
    for i in range(iterations):
        rand_index = np.random.randint(0, m)
        X_i = X[rand_index, :].reshape(1, -1)
        y_i = y[rand_index]
        gradient = (1 / m) * X_i.T.dot(X_i.dot(weights) - y_i)
        weights = weights - learning_rate * gradient
        cost = cost_function(X, y, weights)
        cost_history.append(cost)
        if abs(prev_cost - cost) < tolerance:
            print(f'Convergence achieved at iteration {i+1} with learning rate {learning_rate}')
            break
        prev_cost = cost
    return weights, cost_history

# Tune the learning rate
learning_rates = [1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]
best_lr = 0
best_weights = None
best_cost = float('inf')
max_cost_history = []

for lr in learning_rates:
    weights, cost_history = stochastic_gradient_descent(X_train, y_train, learning_rate=lr)
    final_cost = cost_history[-1]
    if final_cost < best_cost:
        best_lr = lr
        best_weights = weights
        best_cost = final_cost
        max_cost_history = cost_history if len(cost_history) > len(max_cost_history) else max_cost_history

    print(f'Learning Rate: {lr}, Final Cost: {final_cost}, Number of Iterations: {len(cost_history)}')

# Report the learned weight vector and learning rate
print(f'Best Learning Rate: {best_lr}')
print(f'Learned SGD Weight Vector: {best_weights}')

# Plot the cost function value against the number of iterations
plt.figure(figsize=(14,6))
plt.plot(range(len(max_cost_history)), max_cost_history)
plt.xlabel('Number of Iterations')
plt.ylabel('Cost Function Value')
plt.title('Cost Function Value vs Number of Iterations for SGD')
plt.savefig('sgd.png') 
plt.show()

# Evaluate the model on the test data
test_cost = cost_function(X_test, y_test, best_weights)
print(f'Test Cost: {test_cost}')

# Calculate the optimal weight vector analytically
optimal_weights = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)

# Report the optimal weight vector
print(f'Optimal Weight Vector (Analytical): {optimal_weights}')

    