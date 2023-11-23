from scipy.optimize import minimize
import numpy as np
import pandas as pd

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

# Define the kernel function
# def linear_kernel_matrix(X):
#     return np.dot(X, X.T)
def linear_kernel_matrix(X1, X2=None):
    if X2 is None:
        X2 = X1
    return np.dot(X1, X2.T)

# Compute the kernel matrix K
K = linear_kernel_matrix(X_train)

# Define the objective function for the dual SVM
def dual_objective_function(alpha, X, y):
    # Calculate the dual objective function value
    return np.sum(alpha) - 0.5 * np.sum(alpha[:, None] * alpha[None, :] * y[:, None] * y[None, :] * K)

# Define constraints and bounds for the optimization
def zero_constraint(alpha, y):
    return np.dot(alpha, y)

# The constraint is that the sum of alpha * y should be 0
cons = ({'type': 'eq', 'fun': zero_constraint, 'args': (y_train,)})

# Bounds for alpha_i: 0 <= alpha_i <= C
def bounds(C, size):
    return [(0, C) for _ in range(size)]

# Implement the SVM in the dual domain
C_values = [100/873, 500/873, 700/873]
for C in C_values:
    initial_alpha = np.zeros(len(y_train))
    bnds = bounds(C, len(y_train))
    
    # Run the optimization to get alphas
    result = minimize(lambda a: -dual_objective_function(a, X_train, y_train),
                      initial_alpha, bounds=bnds, constraints=cons, method='SLSQP')
    alphas = result.x

    # Compute the weights w
    w = np.sum((alphas * y_train)[:, None] * X_train, axis=0)
    
    # Compute the bias b using support vectors (alphas > 0)
    # support_vectors_indices = alphas > 1e-5
    # support_vectors = X_train[support_vectors_indices]
    # support_vector_labels = y_train[support_vectors_indices]
    # support_vector_alphas = alphas[support_vectors_indices]
    # b = np.mean(support_vector_labels - np.dot(support_vectors, w))
    # support_vectors_indices = (alphas > 1e-5) & (alphas < C - 1e-5)
    # b = np.mean([y_train[i] - np.sum(alphas * y_train * K[i, support_vectors_indices])
    #              for i in range(len(y_train)) if support_vectors_indices[i]])
    
    support_vectors_indices = (alphas > 1e-5) & (alphas < C - 1e-5)
    support_vectors_alphas = alphas[support_vectors_indices]
    support_vectors = X_train[support_vectors_indices]
    support_vector_labels = y_train[support_vectors_indices]

    # Recompute the kernel matrix K for only the support vectors
    support_vectors_kernel_matrix = linear_kernel_matrix(support_vectors)

    # Calculate the bias term b
    b = np.mean([
        y_train[i] - np.sum(support_vectors_alphas * support_vector_labels * support_vectors_kernel_matrix[:, i])
        for i in range(len(support_vector_labels))
    ])

    
    # Output the learned w and b
    print(f"Learned weights for C={C}: {w}")
    print(f"Learned bias for C={C}: {b}")
    
    # Make predictions on the training and test data
    train_kernel = linear_kernel_matrix(X_train)
    test_kernel = linear_kernel_matrix(X_test, support_vectors)
    train_predictions = np.sign(np.dot(train_kernel, alphas * y_train) + b)
    test_predictions = np.sign(np.dot(test_kernel, support_vectors_alphas * support_vector_labels) + b)
    train_error = np.mean(train_predictions != y_train)
    test_error = np.mean(test_predictions != y_test)
    
    # Output the training and test errors
    print(f"Training error for C={C}: {train_error}")
    print(f"Test error for C={C}: {test_error}\n")

# Define the Gaussian kernel function
def gaussian_kernel_matrix(X1, X2, gamma):
    # Compute the squared Euclidean distance matrix
    sq_dist = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * sq_dist)

# Test different values of C and gamma
C_values = [100/873, 500/873, 700/873]
gamma_values = [0.1, 0.5, 1, 5, 100]

for C in C_values:
    for gamma in gamma_values:
        # Compute the kernel matrix K using the Gaussian kernel
        K = gaussian_kernel_matrix(X_train, X_train, gamma)

        # Redefine the dual objective function to use the new kernel matrix
        def dual_objective_function(alpha):
            return np.sum(alpha) - 0.5 * np.sum(alpha[:, None] * alpha[None, :] * y_train[:, None] * y_train[None, :] * K)

        # Run the optimization to get alphas
        initial_alpha = np.zeros(len(y_train))
        bnds = bounds(C, len(y_train))
        result = minimize(lambda a: -dual_objective_function(a),
                          initial_alpha, bounds=bnds, constraints=cons, method='SLSQP')
        alphas = result.x

        # Compute the bias b using support vectors
        support_vectors_indices = (alphas > 1e-5) & (alphas < C - 1e-5)
        support_vectors_alphas = alphas[support_vectors_indices]
        support_vectors = X_train[support_vectors_indices]
        support_vector_labels = y_train[support_vectors_indices]
        support_vectors_kernel_matrix = gaussian_kernel_matrix(support_vectors, support_vectors, gamma)
        b = np.mean([y_train[i] - np.sum(support_vectors_alphas * support_vector_labels * support_vectors_kernel_matrix[:, i])
                     for i in range(len(support_vector_labels))])

        # Make predictions on the training and test data
        train_kernel = gaussian_kernel_matrix(X_train, support_vectors, gamma)
        test_kernel = gaussian_kernel_matrix(X_test, support_vectors, gamma)
        train_predictions = np.sign(np.dot(train_kernel, support_vectors_alphas * support_vector_labels) + b)
        test_predictions = np.sign(np.dot(test_kernel, support_vectors_alphas * support_vector_labels) + b)

        # Compute errors
        train_error = np.mean(train_predictions != y_train)
        test_error = np.mean(test_predictions != y_test)

        # Output the errors
        print(f"C={C}, gamma={gamma}, train_error={train_error}, test_error={test_error}")

def compare_support_vectors(support_vectors_dict, C):
    gamma_values = list(support_vectors_dict[C].keys())
    overlap_counts = {}

    for i in range(len(gamma_values) - 1):
        gamma1 = gamma_values[i]
        gamma2 = gamma_values[i + 1]
        sv1 = support_vectors_dict[C][gamma1]
        sv2 = support_vectors_dict[C][gamma2]
        overlap = len(sv1.intersection(sv2))
        overlap_counts[(gamma1, gamma2)] = overlap

    return overlap_counts

# Store the support vectors for each C and gamma
support_vectors_dict = {C: {} for C in C_values}

C_values = [100/873, 500/873, 700/873]
gamma_values = [0.1, 0.5, 1, 5, 100]

for C in C_values:
    for gamma in gamma_values:
        # Compute the Gaussian kernel matrix
        K = gaussian_kernel_matrix(X_train, X_train, gamma)

        # Bounds and initial alphas
        bnds = bounds(C, len(y_train))
        initial_alpha = np.zeros(len(y_train))

        # Run the optimization
        result = minimize(lambda a: -dual_objective_function(a, K, y_train),
                          initial_alpha, bounds=bnds, constraints=cons, method='SLSQP')
        alphas = result.x

        # Identify and store support vectors
        support_vectors_indices = np.where((alphas > 1e-5) & (alphas < C - 1e-5))[0]
        support_vectors_alphas = alphas[support_vectors_indices]
        support_vectors = X_train[support_vectors_indices]
        support_vector_labels = y_train[support_vectors_indices]

        # Corrected bias calculation
        b_sum = 0
        for i in range(len(support_vectors)):
            kernel_result = np.sum(support_vectors_alphas * support_vector_labels * np.exp(-gamma * np.linalg.norm(support_vectors - support_vectors[i], axis=1) ** 2))
            b_sum += support_vector_labels[i] - kernel_result

        b = b_sum / len(support_vectors) if len(support_vectors) > 0 else 0
        
        # Store the support vectors
        support_vectors_dict[C][gamma] = set(support_vectors_indices)

        # Make predictions
        train_kernel = gaussian_kernel_matrix(X_train, support_vectors, gamma)
        test_kernel = gaussian_kernel_matrix(X_test, support_vectors, gamma)
        train_predictions = np.sign(np.dot(train_kernel, support_vectors_alphas * support_vector_labels) + b)
        test_predictions = np.sign(np.dot(test_kernel, support_vectors_alphas * support_vector_labels) + b)

        # Compute errors
        train_error = np.mean(train_predictions != y_train)
        test_error = np.mean(test_predictions != y_test)

        # Output the errors
        print(f"C={C}, gamma={gamma}, train_error={train_error}, test_error={test_error}")

# Compare support vectors for C = 500/873
C_specific = 500/873
overlap_counts = compare_support_vectors(support_vectors_dict, C_specific)

# Output the number of support vectors and their overlaps
for C in C_values:
    print(f"Number of support vectors for C={C}:")
    for gamma in gamma_values:
        num_sv = len(support_vectors_dict[C][gamma])
        print(f"  gamma={gamma}: {num_sv}")
    print()

print(f"Overlaps of support vectors when C={C_specific}:")
for pair, count in overlap_counts.items():
    print(f"  Gamma {pair[0]} to {pair[1]}: {count}")