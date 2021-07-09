import numpy as np

class LinearRegression:

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
        for i in range(self.num_iterations):
            y_pred = self.predict(X)
            
            # calculating gradients
            dweights = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            dbias = (1 / n_samples) * np.sum(y_pred - y)
            
            # updating parameters
            self.weights -= self.learning_rate * dweights
            self.bias -= self.learning_rate * dbias


    def mse(self, y_pred, y):
        # calculates mean squared error
        mse = np.mean((y_pred - y) ** 2)
        return mse


    def predict(self, X):
        # linear regression
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred