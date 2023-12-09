import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim

# Load data
train_data = pd.read_csv('data/train.csv', header=None)
test_data = pd.read_csv('data/test.csv', header=None)

# Split features and labels
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train, y_train = torch.Tensor(X_train), torch.LongTensor(y_train)
X_test, y_test = torch.Tensor(X_test), torch.LongTensor(y_test)

# Create data loaders
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

input_size = X_train.shape[1]
output_size = len(torch.unique(y_train))

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, depth, activation_func):
        super(NeuralNetwork, self).__init__()
        layers = [nn.Linear(input_size, hidden_size), activation_func()]
        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation_func())
        layers.append(nn.Linear(hidden_size, output_size))
        self.layers = nn.Sequential(*layers)
        self._init_weights(activation_func)

    def _init_weights(self, activation_func):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if activation_func == nn.Tanh:
                    nn.init.xavier_normal_(m.weight)
                elif activation_func == nn.ReLU:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        return self.layers(x)

def train_and_evaluate(model, train_loader, test_loader, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        train_error, test_error = 0, 0
        # Calculate training error
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, targets in train_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            train_error = 1 - correct / total

        # Calculate test error
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, targets in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            test_error = 1 - correct / total

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}, Train Error: {train_error}, Test Error: {test_error}")

    return train_error, test_error

depths = [3, 5, 9]
widths = [5, 10, 25, 50, 100]
for depth in depths:
    for width in widths:
        for activation_func in [nn.Tanh, nn.ReLU]:
            model = NeuralNetwork(input_size, width, output_size, depth, activation_func)
            train_error, test_error = train_and_evaluate(model, train_loader, test_loader)
            print(f"Depth: {depth}, Width: {width}, Activation: {activation_func.__name__}, Train Error: {train_error}, Test Error: {test_error}")
