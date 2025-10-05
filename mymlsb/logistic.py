import numpy as np
from mymlsb.metrics import bce

def sigmoid(z):
    """
    Sigmoid function
    """

    return 1 / (1 + np.exp(-z))

class LogisticRegressionWithGD():
    """
    """

    def __init__(self, lr=0.001, max_iter=1000):
        """
        """
        
        self.lr = lr
        self.max_iter = max_iter
        self.weights = None
        self.bias = None

        return
    
    def fit(self, X, y):
        """
        """

        m, n = X.shape
        weights = np.random.normal(size=n, loc=0, scale=0.01)
        bias = 0.0
        performance = np.full(self.max_iter, np.nan)
        for i_iter in range(self.max_iter):
            z = X @ weights + bias
            h = sigmoid(z)
            loss = bce(h, y)
            performance[i_iter] = loss
            dw = 1 / m * (X.T @ (h - y))
            db = np.mean(h - y)
            weights -= (self.lr * dw)
            bias -= (self.lr * db)

        #
        self.weights = weights
        self.bias = bias

        return performance
    
    def predict(self, X):
        """
        """

        proba = self.predict_proba(X)

        return np.array([0 if proba[i] < 0.5 else 1 for i in range(len(proba))])
    
    def predict_proba(self, X):
        """
        """

        z = X @ self.weights + self.bias

        return sigmoid(z)