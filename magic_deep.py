import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

# Fetch Magic Gamma Telescope dataset
data = fetch_ucirepo(id=159)
X = data.data.features
y = data.data.targets

# One-hot encode target labels (since it's a binary classification problem)
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
def deep_rvfl(X_train, Y_train, layers, C, X_test, Y_test):
    np.random.seed(42)  # For reproducibility

    # Dimensions
    N, d = X_train.shape

    # Initialize weights and biases for each hidden layer
    weights = [np.random.uniform(-1, 1, (d if i == 0 else layers[i-1], layers[i])) for i in range(len(layers))]
    biases = [np.random.uniform(0, 1, (l,)) for l in layers]

    # Forward pass through all layers
    H = X_train
    for i in range(len(layers)):
        H = np.tanh(np.dot(H, weights[i]) + biases[i])

    # Concatenate original input and all hidden layers outputs
    H_all = np.concatenate([X_train] + [H], axis=1)

    # Compute output weights (beta)
    if d + sum(layers) <= N:
        beta = np.dot(np.linalg.pinv(np.dot(H_all.T, H_all) + C * np.eye(H_all.shape[1])), np.dot(H_all.T, Y_train))
    else:
        beta = np.dot(np.dot(H_all.T, np.linalg.pinv(np.dot(H_all, H_all.T) + C * np.eye(H_all.shape[0]))), Y_train)

    # Testing Phase
    H_test = X_test
    for i in range(len(layers)):
        H_test = np.tanh(np.dot(H_test, weights[i]) + biases[i])

    H_test_all = np.concatenate([X_test] + [H_test], axis=1)
    Y_pred = np.dot(H_test_all, beta)
    Y_pred_labels = np.argmax(Y_pred, axis=1)

    # Accuracy Calculation
    Y_test_labels = np.argmax(Y_test, axis=1)
    acc = accuracy_score(Y_test_labels, Y_pred_labels)

    return acc


# Hyperparameter search for best layers and C (regularization)
layer_sizes = [128, 256, 512]  # Layer sizes for deep RVFL
C_values = [2 ** x for x in range(6, -13, -2)]  # Regularization values: 2^6 to 2^-12

best_acc = 0
best_layers, best_C = None, None

for layers in [[l] * 3 for l in layer_sizes]:  # Deep RVFL with the same layer size for each hidden layer
    for C in C_values:
        acc = deep_rvfl(X_train, Y_train, layers, C, X_val, Y_val)
        if acc > best_acc:
            best_acc = acc
            best_layers, best_C = layers, C

# Final evaluation on the test set
final_acc = deep_rvfl(X_train_full, Y_train_full, best_layers, best_C, X_test, Y_test)

# Print Results
print(f"Best Validation Accuracy: {best_acc:.4f}")
print(f"Best Layers (hidden neurons per layer): {best_layers}")
print(f"Best C (regularization): {best_C}")
print(f"Test Accuracy: {final_acc:.4f}")
