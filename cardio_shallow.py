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


# Define Shallow RVFL Model with Hyperparameter Tuning
class ShallowRVFL:
    def __init__(self, num_hidden_neurons=1024, lambda_reg=0.1):
        self.num_hidden_neurons = num_hidden_neurons
        self.lambda_reg = lambda_reg

    def fit(self, X_train, y_train):
        # Randomly initialize input-to-hidden weights
        self.W = np.random.randn(X_train.shape[1], self.num_hidden_neurons)
        self.b = np.random.randn(self.num_hidden_neurons)

        # Compute hidden layer output using tanh activation
        H = np.tanh(X_train.dot(self.W) + self.b)

        # Concatenate original features with hidden layer output
        H_aug = np.hstack([X_train, H])

        # Compute output weights using ridge regression with pseudo-inverse for stability
        self.W_out = np.linalg.pinv(H_aug.T @ H_aug + self.lambda_reg * np.eye(H_aug.shape[1])) @ H_aug.T @ y_train

    def predict(self, X):
        H = np.tanh(X.dot(self.W) + self.b)
        H_aug = np.hstack([X, H])

        y_pred = H_aug.dot(self.W_out)
        return np.argmax(y_pred, axis=1) if len(y_pred.shape) > 1 else (y_pred > 0.5).astype(int)


# Hyperparameter Search
best_accuracy = 0
best_L = None
best_C = None

L_values = [256, 512, 1024]  # Hidden neurons
C_values = [2**x for x in range(-12, 6, 2)]  # Regularization parameter

for L in L_values:
    for C in C_values:
        shallow_rvfl = ShallowRVFL(num_hidden_neurons=L, lambda_reg=1/C)
        shallow_rvfl.fit(X_train, y_train)
        y_pred = shallow_rvfl.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"L = {L}, C = {C}, Accuracy: {acc * 100:.2f}%")

        if acc > best_accuracy:
            best_accuracy = acc
            best_L = L
            best_C = C

print(f"\n Best L: {best_L}, Best C: {best_C}, Best Accuracy: {best_accuracy * 100:.2f}% ")
