import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate = 0.01, num_iterations = 1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None


    def fit(self, X, y):
        # getting no of examples and features
        n_samples, n_features = X.shape

        # initializing weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)

            # calculating gradients
            dweights = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            dbias = (1 / n_samples) * np.sum(y_pred - y)

            # updating parameters
            self.weights -= self.learning_rate * dweights
            self.bias -= self.learning_rate * dbias


    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)
        y_pred_cls = np.where(y_pred > 0.5, 1, 0)
        return y_pred_cls


    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


    def accuracy_score(self, y_pred, y):
        accuracy = np.sum(y == y_pred) / len(y)
        return accuracy