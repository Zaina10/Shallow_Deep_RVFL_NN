import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer

#  Step 1: Load dataset
car_evaluation = fetch_ucirepo(id=19)
X = car_evaluation.data.features
y = car_evaluation.data.targets

#  Step 2: Encode categorical features (One-Hot Encoding)
column_transformer = ColumnTransformer(
    [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), X.columns)],
    remainder="passthrough"
)
X_encoded = column_transformer.fit_transform(X)

#  Step 3: Encode target labels (One-Hot Encoding for Multi-Class)
le = LabelEncoder()
y_encoded = le.fit_transform(y.values.ravel())  # Convert labels to numeric
onehot_encoder = OneHotEncoder(sparse_output=False)
y_onehot = onehot_encoder.fit_transform(y_encoded.reshape(-1, 1))  # One-hot encoding

#  Step 4: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_onehot, test_size=0.2, random_state=42)

#  Step 5: Standardize Data
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
        input_size = X_train.shape[1]
        H = X_train  # Start with input data

        # Generate random weights and biases for multiple hidden layers
        for i in range(self.num_hidden_layers):
            W = np.random.randn(H.shape[1], self.num_hidden_neurons)  # Fix: Match previous layer size
            b = np.random.randn(self.num_hidden_neurons)
            self.hidden_weights.append(W)
            self.hidden_biases.append(b)
            H = np.tanh(H.dot(W) + b)  # Apply activation

        # Concatenate direct links (skip connections)
        H = np.hstack([X_train, H])

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

        #  Multi-Class Fix: Use `argmax()`
        return np.argmax(y_pred, axis=1)


#  Initialize and train Deep RVFL
deep_rvfl = DeepRVFL(num_hidden_layers=3, num_hidden_neurons=512, lambda_reg=0.1)
deep_rvfl.fit(X_train, y_train)

# Predictions
y_pred = deep_rvfl.predict(X_test)

# Convert one-hot encoded y_test back to class indices
y_test_labels = np.argmax(y_test, axis=1)

#  Accuracy
accuracy = accuracy_score(y_test_labels, y_pred)
print(f" Deep RVFL Multi-Class Accuracy: {accuracy * 100:.2f}%")
