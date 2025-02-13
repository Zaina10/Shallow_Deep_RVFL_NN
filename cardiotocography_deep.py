import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# Fetch dataset
cardiotocography = fetch_ucirepo(id=193)

# Extract features and target (NSP column)
X = cardiotocography.data.features
y = cardiotocography.data.targets['NSP']

# Encode target labels (if categorical)
le = LabelEncoder()
y = le.fit_transform(y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Define Deep RVFL Model with Hyperparameter Tuning
class DeepRVFL:
    def __init__(self, num_hidden_neurons=[1024, 512], lambda_reg=0.1):
        self.num_hidden_neurons = num_hidden_neurons  # List of neurons per layer
        self.lambda_reg = lambda_reg  # Regularization parameter

    def fit(self, X_train, y_train):
        self.W = []
        self.b = []

        # Randomly initialize input-to-hidden weights for each layer
        for i in range(len(self.num_hidden_neurons)):
            W_layer = np.random.randn(X_train.shape[1] if i == 0 else self.num_hidden_neurons[i-1], self.num_hidden_neurons[i])
            b_layer = np.random.randn(self.num_hidden_neurons[i])
            self.W.append(W_layer)
            self.b.append(b_layer)

        # Forward pass through each hidden layer
        H = X_train
        for i in range(len(self.num_hidden_neurons)):
            H = np.tanh(H.dot(self.W[i]) + self.b[i])

        # Concatenate original features with hidden layer output
        H_aug = np.hstack([X_train, H])

        # Compute output weights using ridge regression with pseudo-inverse for stability
        self.W_out = np.linalg.pinv(H_aug.T @ H_aug + self.lambda_reg * np.eye(H_aug.shape[1])) @ H_aug.T @ y_train

    def predict(self, X):
        # Forward pass through each hidden layer
        H = X
        for i in range(len(self.num_hidden_neurons)):
            H = np.tanh(H.dot(self.W[i]) + self.b[i])

        # Concatenate original features with hidden layer output
        H_aug = np.hstack([X, H])

        y_pred = H_aug.dot(self.W_out)
        return np.argmax(y_pred, axis=1) if len(y_pred.shape) > 1 else (y_pred > 0.5).astype(int)


# Hyperparameter Search
best_accuracy = 0
best_L = None
best_C = None

L_values = [256, 512, 1024]  # Hidden neurons for each layer (you can add more layers with different sizes)
C_values = [2**x for x in range(-12, 6, 2)]  # Regularization parameter

# Perform hyperparameter search over L and C
for L in L_values:
    for C in C_values:
        deep_rvfl = DeepRVFL(num_hidden_neurons=[L, L], lambda_reg=1/C)  # 2 layers with L neurons
        deep_rvfl.fit(X_train, y_train)
        y_pred = deep_rvfl.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"L = {L}, C = {C}, Accuracy: {acc * 100:.2f}%")

        if acc > best_accuracy:
            best_accuracy = acc
            best_L = L
            best_C = C

# Print Best Hyperparameters
print(f"\nBest L: {best_L}, Best C: {best_C}, Best Accuracy: {best_accuracy * 100:.2f}%")
