import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo

#  Step 1: Load dataset
page_blocks_classification = fetch_ucirepo(id=78)
X = page_blocks_classification.data.features
y = page_blocks_classification.data.targets

#  Step 2: Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y.values.ravel())  # Convert to numerical labels

#  Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

#  Step 4: Standardize Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#  Define Deep RVFL Model
class DeepRVFL:
    def __init__(self, num_hidden_layers=3, num_hidden_neurons=512, lambda_reg=0.1):
        self.num_hidden_layers = num_hidden_layers  # Number of hidden layers
        self.num_hidden_neurons = num_hidden_neurons  # Number of neurons per layer
        self.lambda_reg = lambda_reg  # Regularization parameter
        self.hidden_weights = []  # Store weights for each layer
        self.hidden_biases = []  # Store biases for each layer

    def fit(self, X_train, y_train):
        H = X_train  # Start with input data

        # Generate random weights and biases for multiple hidden layers
        for _ in range(self.num_hidden_layers):
            W = np.random.randn(H.shape[1], self.num_hidden_neurons)  # Match previous layer size
            b = np.random.randn(self.num_hidden_neurons)
            self.hidden_weights.append(W)
            self.hidden_biases.append(b)
            H = np.tanh(H.dot(W) + b)  # Apply activation

        # Concatenate direct links (skip connections)
        H = np.hstack([X_train, H])

        #  Fix: Ensure `y_train` is reshaped to 2D
        y_train = y_train.reshape(-1, 1)

        # Compute output weights using ridge regression
        self.W_out = np.linalg.inv(H.T.dot(H) + self.lambda_reg * np.eye(H.shape[1])).dot(H.T).dot(y_train)

    def predict(self, X):
        H = X

        # Forward pass through hidden layers
        for W, b in zip(self.hidden_weights, self.hidden_biases):
            H = np.tanh(H.dot(W) + b)

        # Concatenate direct links (skip connections)
        H = np.hstack([X, H])

        # Compute final output
        y_pred = H.dot(self.W_out)

        #  Fix: Convert to 1D array
        y_pred = y_pred.flatten()

        return np.round(y_pred).astype(int)  # Convert to class labels


#  Initialize and train Deep RVFL
deep_rvfl = DeepRVFL(num_hidden_layers=3, num_hidden_neurons=512, lambda_reg=0.1)
deep_rvfl.fit(X_train, y_train)

#  Predictions
y_pred = deep_rvfl.predict(X_test)

#  Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f" Deep RVFL Accuracy: {accuracy * 100:.2f}%")
