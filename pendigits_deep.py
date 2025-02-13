import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

# Fetch Pen-based Recognition of Handwritten Digits dataset
data = fetch_ucirepo(id=81)
X = data.data.features
y = data.data.targets

# Print dataset metadata and variable information
print(data.metadata)
print(data.variables)

# One-hot encode target labels (since it's a multi-class problem)
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.values.reshape(-1, 1))

# Standardize features for better training stability
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset into train (70%), validation (20%), and test (10%)
X_train_full, X_test, Y_train_full, Y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_full, Y_train_full, test_size=0.2222,
                                                  random_state=42)  # 20% of total

# Deep RVFL Function
def deep_rvfl(X_train, Y_train, L_values, C, X_test, Y_test):
    np.random.seed(42)  # For reproducibility

    # Dimensions
    N, d = X_train.shape
    layers = len(L_values)

    # Initialize weights and biases for each hidden layer
    weights = [np.random.uniform(-1, 1, (d if i == 0 else L_values[i-1], L_values[i])) for i in range(layers)]
    biases = [np.random.uniform(0, 1, (L_values[i],)) for i in range(layers)]

    # Forward pass through the layers
    H = X_train
    hidden_outputs = []

    for i in range(layers):
        H2 = np.tanh(np.dot(H, weights[i]) + biases[i])  # Activation function
        hidden_outputs.append(H2)
        H = H2  # Output of current layer becomes input to next

    # Concatenate the original input and all hidden layer outputs
    H_combined = np.concatenate([X_train] + hidden_outputs, axis=1)

    # Compute output weights (beta)
    if d + sum(L_values) <= N:
        beta = np.dot(np.linalg.pinv(np.dot(H_combined.T, H_combined) + C * np.eye(H_combined.shape[1])), np.dot(H_combined.T, Y_train))
    else:
        beta = np.dot(np.dot(H_combined.T, np.linalg.pinv(np.dot(H_combined, H_combined.T) + C * np.eye(H_combined.shape[0]))), Y_train)

    # Testing Phase
    H_test = X_test
    hidden_outputs_test = []

    for i in range(layers):
        H2_test = np.tanh(np.dot(H_test, weights[i]) + biases[i])  # Activation function
        hidden_outputs_test.append(H2_test)
        H_test = H2_test  # Output of current layer becomes input to next

    # Concatenate the original input and all hidden layer outputs
    H_test_combined = np.concatenate([X_test] + hidden_outputs_test, axis=1)

    # Prediction
    Y_pred = np.dot(H_test_combined, beta)
    Y_pred_labels = np.argmax(Y_pred, axis=1)

    # Accuracy Calculation
    Y_test_labels = np.argmax(Y_test, axis=1)
    acc = accuracy_score(Y_test_labels, Y_pred_labels)

    return acc


# Hyperparameter search for best L (hidden neurons) for each layer and C (regularization)
L_values = [256, 512, 1024]  # Adjusted based on dataset size for each layer
C_values = [2 ** x for x in range(6, -13, -2)]  # Regularization values: 2^6 to 2^-12

best_acc = 0
best_Ls, best_C = None, None

for Ls in [[256, 512], [512, 1024], [1024, 2048]]:
    for C in C_values:
        acc = deep_rvfl(X_train, Y_train, Ls, C, X_val, Y_val)
        if acc > best_acc:
            best_acc = acc
            best_Ls, best_C = Ls, C

# Final evaluation on the test set
final_acc = deep_rvfl(X_train_full, Y_train_full, best_Ls, best_C, X_test, Y_test)

# Print Results
print(f"Best Validation Accuracy: {best_acc:.4f}")
print(f"Best L (hidden neurons per layer): {best_Ls}")
print(f"Best C (regularization): {best_C}")
print(f"Test Accuracy: {final_acc:.4f}")
