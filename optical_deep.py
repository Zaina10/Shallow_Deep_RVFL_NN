import numpy as np
import pandas as pd
import ucimlrepo
# import scikit learn as sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo

# Fix random seed for reproducibility
np.random.seed(42)

# Fetch dataset
dataset = fetch_ucirepo(id=80)
X = dataset.data.features
y = dataset.data.targets

# One-hot encode target variable
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.values.reshape(-1, 1))

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-Test split
X_train_full, X_test, Y_train_full, Y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Further split training data into training (50%) and validation (20%)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_full, Y_train_full, test_size=0.2857, random_state=42)

# Deep RVFL Function
def deep_rvfl(X_train, Y_train, X_test, Y_test, L, C, depth):
    np.random.seed(42)  # Ensure reproducibility
    N, d = X_train.shape  # Samples, features

    # Initialize hidden layer transformations
    H_train = X_train
    H_test = X_test

    for _ in range(depth):
        # Generate random weights and biases
        weights = np.random.uniform(-1, 1, (H_train.shape[1], L))
        biases = np.random.uniform(0, 1, (L,))

        # Apply transformation
        H2_train = np.tanh(np.dot(H_train, weights) + biases)
        H_train = np.concatenate([H_train, H2_train], axis=1)

        H2_test = np.tanh(np.dot(H_test, weights) + biases)
        H_test = np.concatenate([H_test, H2_test], axis=1)

    # Compute output weights (beta)
    if H_train.shape[1] <= N:
        beta = np.dot(np.linalg.pinv(np.dot(H_train.T, H_train) + C * np.eye(H_train.shape[1])), np.dot(H_train.T, Y_train))
    else:
        beta = np.dot(np.dot(H_train.T, np.linalg.pinv(np.dot(H_train, H_train.T) + C * np.eye(H_train.shape[0]))), Y_train)

    # Predict and evaluate
    Y_pred = np.dot(H_test, beta)
    Y_pred_labels = np.argmax(Y_pred, axis=1)
    Y_test_labels = np.argmax(Y_test, axis=1)

    acc = accuracy_score(Y_test_labels, Y_pred_labels)
    return acc

# Hyperparameter search
L_values = [256, 512, 1024]  # Hidden neurons
C_values = [2**x for x in range(6, -12, -2)]  # Regularization
depth_values = [1, 2, 3, 4, 5]  # Depth levels

best_acc = 0
best_L, best_C, best_depth = None, None, None

for L in L_values:
    for C in C_values:
        for depth in depth_values:
            acc = deep_rvfl(X_train, Y_train, X_val, Y_val, L, C, depth)
            if acc > best_acc:
                best_acc = acc
                best_L = L
                best_C = C
                best_depth = depth

# Final evaluation on the test set
final_acc = deep_rvfl(X_train_full, Y_train_full, X_test, Y_test, best_L, best_C, best_depth)

print(f"Best Validation Accuracy: {best_acc}")
print(f"Best L (hidden neurons): {best_L}")
print(f"Best C (regularization): {best_C}")
print(f"Best Depth: {best_depth}")
print(f"Test Accuracy: {final_acc}")
