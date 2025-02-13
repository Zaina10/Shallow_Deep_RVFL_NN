import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score

# Fetch dataset (UCI Mushroom dataset)
mushroom = fetch_openml(data_id=73, as_frame=True)
X = mushroom.data
y = mushroom.target

# Convert categorical features using OneHotEncoding (dense matrix)
column_transformer = ColumnTransformer(
    [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), X.columns)],
    # Set sparse_output=False to get a dense matrix
    remainder="passthrough"
)

# Apply column transformation (one-hot encoding)
X_encoded = column_transformer.fit_transform(X)

# Encode target labels (mushroom target values are categorical)
le = LabelEncoder()
y = le.fit_transform(y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Standardize data (now all numeric)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Define Shallow RVFL
class ShallowRVFL:
    def __init__(self, num_hidden_neurons=1024, lambda_reg=0.1):
        self.num_hidden_neurons = num_hidden_neurons
        self.lambda_reg = lambda_reg

    def fit(self, X_train, y_train):
        # Random initialization of weights (input to hidden layer)
        self.W_in = np.random.randn(X_train.shape[1], self.num_hidden_neurons)
        self.b = np.random.randn(self.num_hidden_neurons)  # Bias for each hidden neuron

        # Compute the hidden layer output (non-linear activation can be used)
        H_train = np.tanh(X_train.dot(self.W_in) + self.b)  # tanh activation
        # Compute the output weights using ridge regression
        H_train = np.hstack([H_train, np.ones((H_train.shape[0], 1))])  # Add bias column
        self.W_out = np.linalg.inv(H_train.T.dot(H_train) + self.lambda_reg * np.eye(H_train.shape[1])).dot(
            H_train.T).dot(y_train)

    def predict(self, X):
        # Compute the hidden layer output (same activation)
        H = np.tanh(X.dot(self.W_in) + self.b)
        H = np.hstack([H, np.ones((H.shape[0], 1))])  # Add bias column
        y_pred = H.dot(self.W_out)
        return np.argmax(y_pred, axis=1)  # Predicted class labels


# Initialize and train the Shallow RVFL
model = ShallowRVFL(num_hidden_neurons=1024, lambda_reg=0.1)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

