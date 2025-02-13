import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo

# Fetch Mushroom dataset
mushroom = fetch_ucirepo(id=73)

# Extract features and labels
X = mushroom.data.features
y = mushroom.data.targets

# One-Hot Encode categorical features
column_transformer = ColumnTransformer(
    [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), X.columns)],  # Dense matrix
    remainder="passthrough"
)

X_encoded = column_transformer.fit_transform(X)

# Encode target labels (convert categorical to binary labels)
le = LabelEncoder()
y = le.fit_transform(y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Standardize data (normalize numeric values)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Define Deep RVFL Model
class DeepRVFL:
    def __init__(self, num_layers=3, num_hidden_neurons=1024, lambda_reg=0.1):
        self.num_layers = num_layers  # Number of hidden layers
        self.num_hidden_neurons = num_hidden_neurons  # Neurons per layer
        self.lambda_reg = lambda_reg  # Regularization parameter

    def fit(self, X_train, y_train):
        self.hidden_layers = []  # Store randomly initialized weights for each layer
        self.biases = []  # Store biases for each layer

        # Feature expansion using deep hidden layers
        H = X_train  # Start with original input

        for _ in range(self.num_layers):
            W = np.random.randn(H.shape[1], self.num_hidden_neurons)  # Random weights
            b = np.random.randn(self.num_hidden_neurons)  # Random bias

            # Compute hidden layer output
            H_new = np.tanh(H.dot(W) + b)  # Apply activation function

            # Store weights and biases
            self.hidden_layers.append(W)
            self.biases.append(b)

            # Concatenate with original input (skip connection)
            H = np.hstack([H, H_new])

        # Compute output weights using ridge regression
        self.W_out = np.linalg.inv(H.T.dot(H) + self.lambda_reg * np.eye(H.shape[1])).dot(H.T).dot(y_train)

    def predict(self, X):
        H = X  # Start with original input

        for W, b in zip(self.hidden_layers, self.biases):
            H_new = np.tanh(H.dot(W) + b)  # Compute hidden layer output
            H = np.hstack([H, H_new])  # Concatenate with previous layers

        # Compute final predictions
        y_pred = H.dot(self.W_out)

        return (y_pred > 0.5).astype(int)  # Convert to binary class labels (0 or 1)


# Initialize and train Deep RVFL
deep_rvfl = DeepRVFL(num_layers=3, num_hidden_neurons=1024, lambda_reg=0.1)
deep_rvfl.fit(X_train, y_train)

# Predictions
y_pred = deep_rvfl.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Deep RVFL Accuracy: {accuracy * 100:.2f}%")
