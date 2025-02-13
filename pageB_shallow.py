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

# Define Shallow RVFL Model
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

        #  Fix: Reshape `y_train` to be a 2D column vector
        y_train = y_train.reshape(-1, 1)

        # Compute output weights using ridge regression
        self.W_out = np.linalg.inv(H.T.dot(H) + self.lambda_reg * np.eye(H.shape[1])).dot(H.T).dot(y_train)

    def predict(self, X):
        H = np.tanh(X.dot(self.W) + self.b)
        H = np.hstack([X, H])  # Concatenate with original input
        y_pred = H.dot(self.W_out)  # Compute final output

        #  Fix: Convert single-column predictions to 1D
        y_pred = y_pred.flatten()

        return np.round(y_pred).astype(int)  # Convert to class labels


#  Initialize and train Shallow RVFL
shallow_rvfl = ShallowRVFL(num_hidden_neurons=1024, lambda_reg=0.1)
shallow_rvfl.fit(X_train, y_train)

#  Predictions
y_pred = shallow_rvfl.predict(X_test)

#  Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f" Shallow RVFL Accuracy: {accuracy * 100:.2f}%")
