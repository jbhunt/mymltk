import numpy as np
from mymltk.metrics import bce

def sigmoid(z):
    """
    Sigmoid function
    """

    return 1 / (1 + np.exp(-z))

class LogisticRegressionClassifier():
    """
    """

    def __init__(self, lr=0.001, max_iter=1000, batch_size=None):
        """
        """
        
        self.lr = lr
        self.max_iter = max_iter
        self.weights = None
        self.bias = None
        self.batch_size = batch_size

        return
    
    def fit(self, X, y):
        """
        """

        m, n = X.shape
        weights = np.random.normal(size=n, loc=0, scale=0.01)
        bias = 0.0
        performance = np.full(self.max_iter, np.nan)
        for i_iter in range(self.max_iter):

            #
            sample_indices = np.arange(m)
            if self.batch_size is not None:
                sample_indices = np.random.choice(sample_indices, size=self.batch_size, replace=False)
            b = sample_indices.size

            #
            X_ = X[sample_indices]
            y_ = y[sample_indices]

            #
            z = X_ @ weights + bias
            h = sigmoid(z)
            loss = bce(h, y_)
            performance[i_iter] = loss
            dw = 1 / b * (X_.T @ (h - y_))
            db = np.mean(h - y_)
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