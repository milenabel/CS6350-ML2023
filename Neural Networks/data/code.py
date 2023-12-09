import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize network architecture
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights with random values from a standard Gaussian distribution
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.hidden_size)
        self.W3 = np.random.randn(self.hidden_size, self.output_size)
        
        # Initialize bias terms
        self.b1 = np.random.randn(self.hidden_size)
        self.b2 = np.random.randn(self.hidden_size)
        self.b3 = np.random.randn(self.output_size)
        
    def forward(self, x):
        # Forward propagation through the network
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.y_hat = sigmoid(self.z3)
        
        return self.y_hat
    
    def backward(self, x, y, y_hat):
        # Backward propagation through the network
        self.y_error = y - y_hat
        self.z3_delta = self.y_error * sigmoid_derivative(self.z3)
        
        self.a2_error = self.z3_delta.dot(self.W3.T)
        self.z2_delta = self.a2_error * sigmoid_derivative(self.z2)
        
        self.a1_error = self.z2_delta.dot(self.W2.T)
        self.z1_delta = self.a1_error * sigmoid_derivative(self.z1)
        
        # Gradients for each layer (dC/dW = a(l-1) * delta(l))
        self.W3_gradient = np.outer(self.a2, self.z3_delta)
        self.W2_gradient = np.outer(self.a1, self.z2_delta)
        self.W1_gradient = np.outer(x, self.z1_delta)
        
        # Gradients for bias terms
        self.b3_gradient = self.z3_delta
        self.b2_gradient = self.z2_delta
        self.b1_gradient = self.z1_delta
        
        return (self.W1_gradient, self.W2_gradient, self.W3_gradient,
                self.b1_gradient, self.b2_gradient, self.b3_gradient)
    
    def update_weights(self, gradients, learning_rate):
        # Update the weights with the calculated gradients
        (W1_gradient, W2_gradient, W3_gradient,
         b1_gradient, b2_gradient, b3_gradient) = gradients
        
        self.W1 += learning_rate * W1_gradient
        self.W2 += learning_rate * W2_gradient
        self.W3 += learning_rate * W3_gradient
        
        self.b1 += learning_rate * b1_gradient
        self.b2 += learning_rate * b2_gradient
        self.b3 += learning_rate * b3_gradient
        
    def calculate_error(self, y):
        # Calculate the error (loss) of the prediction
        return 0.5 * sum((y - self.y_hat) ** 2)

# Let's load a single training example from the training data and then implement back-propagation for it
train_data = pd.read_csv('data/train.csv', header=None)
test_data = pd.read_csv('data/test.csv', header=None)

# Extract features and labels from the training data
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values

# Extract features and labels from the test data
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Get the input size based on the feature count and output size for a single binary classification output
input_size = X_train.shape[1]
output_size = 1

# Select a single training example and label
x_single_example = X_train[0]
y_single_example = y_train[0]

# Instantiate the neural network with the actual architecture
nn_example = NeuralNetwork(input_size=input_size, hidden_size=5, output_size=output_size)

# Forward pass for the single example
y_hat_single_example = nn_example.forward(x_single_example)

# Backward pass for the single example
gradients_single_example = nn_example.backward(x_single_example, y_single_example, y_hat_single_example)

# Error for the single example
error_single_example = nn_example.calculate_error(y_single_example)

# y_hat_single_example, gradients_single_example, error_single_example
    
print("Forward pass output:", y_hat_single_example)
print("Gradients:", gradients_single_example)
print("Error:", error_single_example)


# Stochastic Gradient Descent (SGD) implementation
def train_neural_network(nn, X_train, y_train, X_test, y_test, hidden_sizes, gamma_0, d, epochs):
    for hidden_size in hidden_sizes:
        # Initialize neural network with the specified architecture
        nn = NeuralNetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
        learning_rate = gamma_0
        
        # Track the error history
        train_errors = []
        test_errors = []
        
        # Train for a given number of epochs
        for epoch in range(epochs):
            # Shuffle the training data
            permutation = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]
            
            # Update learning rate according to the schedule
            learning_rate = gamma_0 / (1 + (gamma_0 / d) * epoch)
            
            for x, y in zip(X_train_shuffled, y_train_shuffled):
                # Forward pass
                y_hat = nn.forward(x)
                
                # Backward pass
                gradients = nn.backward(x, y, y_hat)
                
                # Update weights
                nn.update_weights(gradients, learning_rate)
            
            # Calculate training and test error for the current epoch
            train_error = np.mean([nn.calculate_error(y) for y in nn.forward(X_train)])
            test_error = np.mean([nn.calculate_error(y) for y in nn.forward(X_test)])
            train_errors.append(train_error)
            test_errors.append(test_error)
            
            # Optional: Implement early stopping or other convergence criteria
            
        # Print the training and test errors for each hidden layer size
        print(f"Hidden layer size: {hidden_size}")
        print(f"Training error: {train_errors[-1]}")
        print(f"Test error: {test_errors[-1]}")

# Define hyperparameters
hidden_sizes = [5, 10, 25, 50, 100]
gamma_0 = 0.00005  # This is an initial guess, you may need to tune this hyperparameter.
d = 0.00003  # This is an initial guess, you may need to tune this hyperparameter.
epochs = 100  # Number of epochs to train

# Train the neural network
train_neural_network(nn_example, X_train, y_train, X_test, y_test, hidden_sizes, gamma_0, d, epochs)


# import numpy as np
# from itertools import product

# def train_neural_network(nn_class, X_train, y_train, X_test, y_test, hidden_sizes, gamma_0_range, d_range, epochs):
#     best_params = {}
#     best_test_error = float('inf')
#     best_objective_curve = []
    
#     # Values to try based on previous results
#     gamma_0_values = [0.0005, 0.0012]
#     d_values = [0.00005, 0.0003, 0.0007]
    
#     # Generate the range of values for gamma_0 and d
# #     gamma_0_values = np.linspace(gamma_0_range[0], gamma_0_range[1], num=4)
# #     d_values = np.linspace(d_range[0], d_range[1], num=4)

#     # Grid search over gamma_0 and d values
#     for hidden_size, gamma_0, d in product(hidden_sizes, gamma_0_values, d_values):
#         nn = nn_class(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
#         learning_rate = gamma_0
#         train_errors = []
#         test_errors = []
#         convergence_curve = []

#         for epoch in range(epochs):
#             # Shuffle the training data
#             permutation = np.random.permutation(len(X_train))
#             X_train_shuffled = X_train[permutation]
#             y_train_shuffled = y_train[permutation]

#             for i, (x, y) in enumerate(zip(X_train_shuffled, y_train_shuffled)):
#                 y_hat = nn.forward(x)
#                 gradients = nn.backward(x, y, y_hat)
#                 nn.update_weights(gradients, learning_rate)
                
#                 # Update learning rate according to the schedule
#                 learning_rate = gamma_0 / (1 + (gamma_0 / d) * (epoch * len(X_train) + i))
#                 convergence_curve.append(nn.calculate_error(y))

#             # Calculate and store training and test error
#             train_error = np.mean([nn.calculate_error(y) for y in nn.forward(X_train)])
#             test_error = np.mean([nn.calculate_error(y) for y in nn.forward(X_test)])
#             train_errors.append(train_error)
#             test_errors.append(test_error)

#         # Check for convergence by examining the objective curve
#         if len(set(convergence_curve[-5:])) == 1 and test_errors[-1] < best_test_error:
#             best_test_error = test_errors[-1]
#             best_params = {'hidden_size': hidden_size, 'gamma_0': gamma_0, 'd': d}
#             best_objective_curve = convergence_curve

#         # Print results for this hyperparameter set
#         print(f"Hidden size: {hidden_size}, gamma_0: {gamma_0:.4f}, d: {d:.4f}, Training error: {train_error:.2f}, Test error: {test_error:.2f}")

#     # Use the best parameters to test and obtain the final model performance
#     nn_final = nn_class(input_size=input_size, hidden_size=best_params['hidden_size'], output_size=output_size)
#     train_errors_final = []
#     test_errors_final = []

#     # Initialize learning rate with the best gamma_0
#     learning_rate = best_params['gamma_0']

#     for epoch in range(epochs):
#         # Shuffle the training data
#         permutation = np.random.permutation(len(X_train))
#         X_train_shuffled = X_train[permutation]
#         y_train_shuffled = y_train[permutation]

#         for x, y in zip(X_train_shuffled, y_train_shuffled):
#             y_hat = nn_final.forward(x)
#             gradients = nn_final.backward(x, y, y_hat)
#             nn_final.update_weights(gradients, learning_rate)

#         # Update learning rate according to the best schedule
#         learning_rate = best_params['gamma_0'] / (1 + (best_params['gamma_0'] / best_params['d']) * epoch)

#         # Calculate training and test error
#         train_error = np.mean([nn_final.calculate_error(y) for y in nn_final.forward(X_train)])
#         test_error = np.mean([nn_final.calculate_error(y) for y in nn_final.forward(X_test)])
#         train_errors_final.append(train_error)
#         test_errors_final.append(test_error)

#     # Evaluate the final trained network on the test set
#     final_test_error = np.mean([nn_final.calculate_error(y) for y in nn_final.forward(X_test)])
#     print(f"Final test error: {final_test_error}")    
#     # Return the best parameters, corresponding error, and objective curve
#     return best_params, best_test_error, best_objective_curve

# # Hyperparameters range to try
# hidden_sizes = [5, 10, 25, 50, 100]
# # gamma_0_range = (0.0005, 0.0025)  # Range of learning rates to try
# # d_range = (0.00001, 0.001)  # Range of decay rates to try
# epochs = 100

# # Train the network and find the best hyperparameters
# # best_params, best_test_error, best_objective_curve = train_neural_network(NeuralNetwork, X_train, y_train, X_test, y_test, hidden_sizes, gamma_0_range, d_range, epochs)

# # Train the network and find the best hyperparameters
# best_params, best_test_error, best_objective_curve = train_neural_network(
#     NeuralNetwork, X_train, y_train, X_test, y_test, hidden_sizes, gamma_0_values, d_values, epochs
# )

# print(f"Best parameters: {best_params}")
# print(f"Best test error: {best_test_error}")

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize network architecture
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights to zero instead of random values from a standard Gaussian distribution
        self.W1 = np.zeros((self.input_size, self.hidden_size))
        self.W2 = np.zeros((self.hidden_size, self.hidden_size))
        self.W3 = np.zeros((self.hidden_size, self.output_size))
        
        # Initialize bias terms to zero
        self.b1 = np.zeros(self.hidden_size)
        self.b2 = np.zeros(self.hidden_size)
        self.b3 = np.zeros(self.output_size)
        
    def forward(self, x):
        # Forward propagation through the network
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.y_hat = sigmoid(self.z3)
        
        return self.y_hat
    
    def backward(self, x, y, y_hat):
        # Backward propagation through the network
        self.y_error = y - y_hat
        self.z3_delta = self.y_error * sigmoid_derivative(self.z3)
        
        self.a2_error = self.z3_delta.dot(self.W3.T)
        self.z2_delta = self.a2_error * sigmoid_derivative(self.z2)
        
        self.a1_error = self.z2_delta.dot(self.W2.T)
        self.z1_delta = self.a1_error * sigmoid_derivative(self.z1)
        
        # Gradients for each layer (dC/dW = a(l-1) * delta(l))
        self.W3_gradient = np.outer(self.a2, self.z3_delta)
        self.W2_gradient = np.outer(self.a1, self.z2_delta)
        self.W1_gradient = np.outer(x, self.z1_delta)
        
        # Gradients for bias terms
        self.b3_gradient = self.z3_delta
        self.b2_gradient = self.z2_delta
        self.b1_gradient = self.z1_delta
        
        return (self.W1_gradient, self.W2_gradient, self.W3_gradient,
                self.b1_gradient, self.b2_gradient, self.b3_gradient)
    
    def update_weights(self, gradients, learning_rate):
        # Update the weights with the calculated gradients
        (W1_gradient, W2_gradient, W3_gradient,
         b1_gradient, b2_gradient, b3_gradient) = gradients
        
        self.W1 += learning_rate * W1_gradient
        self.W2 += learning_rate * W2_gradient
        self.W3 += learning_rate * W3_gradient
        
        self.b1 += learning_rate * b1_gradient
        self.b2 += learning_rate * b2_gradient
        self.b3 += learning_rate * b3_gradient
        
    def calculate_error(self, y):
        # Calculate the error (loss) of the prediction
        return 0.5 * sum((y - self.y_hat) ** 2)


# Stochastic Gradient Descent (SGD) implementation
def train_neural_network(nn, X_train, y_train, X_test, y_test, hidden_sizes, gamma_0, d, epochs):
    for hidden_size in hidden_sizes:
        # Initialize neural network with the specified architecture
        nn = NeuralNetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
        learning_rate = gamma_0
        
        # Track the error history
        train_errors = []
        test_errors = []
        
        # Train for a given number of epochs
        for epoch in range(epochs):
            # Shuffle the training data
            permutation = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]
            
            # Update learning rate according to the schedule
            learning_rate = gamma_0 / (1 + (gamma_0 / d) * epoch)
            
            for x, y in zip(X_train_shuffled, y_train_shuffled):
                # Forward pass
                y_hat = nn.forward(x)
                
                # Backward pass
                gradients = nn.backward(x, y, y_hat)
                
                # Update weights
                nn.update_weights(gradients, learning_rate)
            
            # Calculate training and test error for the current epoch
            train_error = np.mean([nn.calculate_error(y) for y in nn.forward(X_train)])
            test_error = np.mean([nn.calculate_error(y) for y in nn.forward(X_test)])
            train_errors.append(train_error)
            test_errors.append(test_error)
            
            # Optional: Implement early stopping or other convergence criteria
            
        # Print the training and test errors for each hidden layer size
        print(f"Hidden layer size: {hidden_size}")
        print(f"Training error: {train_errors[-1]}")
        print(f"Test error: {test_errors[-1]}")

# Define hyperparameters
hidden_sizes = [5, 10, 25, 50, 100]
gamma_0 = 0.00005  # This is an initial guess, you may need to tune this hyperparameter.
d = 0.00003  # This is an initial guess, you may need to tune this hyperparameter.
epochs = 100  # Number of epochs to train

# Train the neural network
train_neural_network(nn_example, X_train, y_train, X_test, y_test, hidden_sizes, gamma_0, d, epochs)
