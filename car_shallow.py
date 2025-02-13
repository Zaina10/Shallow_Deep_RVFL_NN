import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo

# Fetch Car Evaluation dataset
car_evaluation = fetch_ucirepo(id=19)
X = car_evaluation.data.features
y = car_evaluation.data.targets

# Encode categorical features (One-Hot Encoding for inputs)
from sklearn.compose import ColumnTransformer
column_transformer = ColumnTransformer(
    [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), X.columns)],
    remainder="passthrough"
)
X_encoded = column_transformer.fit_transform(X)

# Encode target labels (One-Hot Encoding for Multi-Class)
le = LabelEncoder()
y_encoded = le.fit_transform(y.values.ravel())  # Convert target labels to numeric
onehot_encoder = OneHotEncoder(sparse_output=False)
y_onehot = onehot_encoder.fit_transform(y_encoded.reshape(-1, 1))  # One-hot encoding

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_onehot, test_size=0.2, random_state=42)

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Define Multi-Class RVFL Model
class ShallowRVFL:
    def __init__(self, num_hidden_neurons=1024, lambda_reg=0.1):
        self.num_hidden_neurons = num_hidden_neurons
        self.lambda_reg = lambda_reg

    def fit(self, X_train, y_train):
        input_size = X_train.shape[1]

        # Randomly initialize weights and biases
        self.W = np.random.randn(input_size, self.num_hidden_neurons)
        self.b = np.random.randn(self.num_hidden_neurons)

        # Compute hidden layer activations
        H = np.tanh(X_train.dot(self.W) + self.b)

        # Concatenate original features with hidden layer (skip connection)
        H = np.hstack([X_train, H])

        # Compute output weights using ridge regression (Multi-Class Fix)
        self.W_out = np.linalg.inv(H.T.dot(H) + self.lambda_reg * np.eye(H.shape[1])).dot(H.T).dot(y_train)

    def predict(self, X):
        H = np.tanh(X.dot(self.W) + self.b)
        H = np.hstack([X, H])  # Concatenate with original input
        y_pred = H.dot(self.W_out)  # Compute final output

        #  Multi-Class Fix: Use `argmax()` instead of thresholding
        return np.argmax(y_pred, axis=1)


# Initialize and train Shallow RVFL
shallow_rvfl = ShallowRVFL(num_hidden_neurons=1024, lambda_reg=0.1)
shallow_rvfl.fit(X_train, y_train)

# Predictions
y_pred = shallow_rvfl.predict(X_test)

# Convert one-hot encoded y_test back to class indices
y_test_labels = np.argmax(y_test, axis=1)

# Accuracy
accuracy = accuracy_score(y_test_labels, y_pred)
print(f" Shallow RVFL Multi-Class Accuracy: {accuracy * 100:.2f}%")
