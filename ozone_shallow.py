import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo

# Fetch Ozone Level Detection dataset
ozone_level_detection = fetch_ucirepo(id=172)

# Extract features and labels
X = ozone_level_detection.data.features
y = ozone_level_detection.data.targets

# Encode target labels (convert categorical to numerical labels)
le = LabelEncoder()
y = le.fit_transform(y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Define Shallow RVFL Model
class ShallowRVFL:
    def __init__(self, num_hidden_neurons=1024, lambda_reg=0.1):
        self.num_hidden_neurons = num_hidden_neurons  # Number of hidden neurons
        self.lambda_reg = lambda_reg  # Regularization parameter

    def fit(self, X_train, y_train):
        input_size = X_train.shape[1]  # Number of input features

        # Randomly initialize weights and biases
        self.W = np.random.randn(input_size, self.num_hidden_neurons)
        self.b = np.random.randn(self.num_hidden_neurons)

        # Compute hidden layer activations
        H = np.tanh(X_train.dot(self.W) + self.b)

        # Concatenate original features with hidden layer (skip connection)
        H = np.hstack([X_train, H])

        # Compute output weights using ridge regression
        self.W_out = np.linalg.inv(H.T.dot(H) + self.lambda_reg * np.eye(H.shape[1])).dot(H.T).dot(y_train)

    def predict(self, X):
        H = np.tanh(X.dot(self.W) + self.b)  # Compute hidden layer activations
        H = np.hstack([X, H])  # Concatenate with original input
        y_pred = H.dot(self.W_out)  # Compute final output
        return (y_pred > 0.5).astype(int)  # Convert to binary class labels


# Initialize and train Shallow RVFL
shallow_rvfl = ShallowRVFL(num_hidden_neurons=1024, lambda_reg=0.1)
shallow_rvfl.fit(X_train, y_train)

# Predictions
y_pred = shallow_rvfl.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Shallow RVFL Accuracy: {accuracy * 100:.2f}%")
