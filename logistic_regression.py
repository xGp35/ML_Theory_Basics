import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


data = load_breast_cancer()
X, y = data.data, data.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def binary_crossentropy(y_true, y_pred):
    epsilon = 1e-9   # avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def predict_proba(X, weights, bias):
    linear_model = np.dot(X, weights) + bias
    return sigmoid(linear_model)

def train(X, y, lr=0.01, epochs=1000):
    n_samples, n_features = X.shape

    weights = np.zeros(n_features)
    bias = 0

    for _ in range(epochs):
        y_pred = predict_proba(X, weights, bias)

        dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
        db = (1 / n_samples) * np.sum(y_pred - y)

        weights -= lr * dw
        bias -= lr * db

    return weights, bias

def predict(X, weights, bias, threshold=0.5):
    probs = predict_proba(X, weights, bias)
    return (probs >= threshold).astype(int)


#🚀 5. Train the Model
weights, bias = train(X_train, y_train, lr=0.01, epochs=2000)

#📊 6. Evaluate
y_probs = predict_proba(X_test, weights, bias)
loss = binary_crossentropy(y_test, y_probs)

y_pred = predict(X_test, weights, bias)
accuracy = accuracy_score(y_test, y_pred)

print("Loss:", loss)
print("Accuracy:", accuracy)