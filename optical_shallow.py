import numpy as np
import pandas as pd
import ucimlrepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo

# Fetch Optical Recognition of Handwritten Digits dataset
data = fetch_ucirepo(id=80)
X = data.data.features
y = data.data.targets

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


# RVFL Function (Shallow RVFL)
def rvfl(X_train, Y_train, L, C, X_test, Y_test):
    np.random.seed(42)  # For reproducibility

    # Dimensions
    N, d = X_train.shape

    # Random weights and biases for hidden neurons
    weights = np.random.uniform(-1, 1, (d, L))
    biases = np.random.uniform(0, 1, (L,))

    # Hidden layer transformation
    H2 = np.tanh(np.dot(X_train, weights) + biases)

    # Concatenate original input and hidden layer output
    H = np.concatenate([X_train, H2], axis=1)

    # Compute output weights (beta)
    if d + L <= N:
        beta = np.dot(np.linalg.pinv(np.dot(H.T, H) + C * np.eye(H.shape[1])), np.dot(H.T, Y_train))
    else:
        beta = np.dot(np.dot(H.T, np.linalg.pinv(np.dot(H, H.T) + C * np.eye(H.shape[0]))), Y_train)

    # Testing Phase
    H2_test = np.tanh(np.dot(X_test, weights) + biases)
    H_test = np.concatenate([X_test, H2_test], axis=1)
    Y_pred = np.dot(H_test, beta)
    Y_pred_labels = np.argmax(Y_pred, axis=1)

    # Accuracy Calculation
    Y_test_labels = np.argmax(Y_test, axis=1)
    acc = accuracy_score(Y_test_labels, Y_pred_labels)

    return acc


# Hyperparameter search for best L (hidden neurons) and C (regularization)
L_values = [256, 512, 1024]  # Adjusted based on dataset size
C_values = [2 ** x for x in range(6, -13, -2)]  # Regularization values: 2^6 to 2^-12

best_acc = 0
best_L, best_C = None, None

for L in L_values:
    for C in C_values:
        acc = rvfl(X_train, Y_train, L, C, X_val, Y_val)
        if acc > best_acc:
            best_acc = acc
            best_L, best_C = L, C

# Final evaluation on the test set
final_acc = rvfl(X_train_full, Y_train_full, best_L, best_C, X_test, Y_test)

# Print Results
print(f"Best Validation Accuracy: {best_acc:.4f}")
print(f"Best L (hidden neurons): {best_L}")
print(f"Best C (regularization): {best_C}")
print(f"Test Accuracy: {final_acc:.4f}")
