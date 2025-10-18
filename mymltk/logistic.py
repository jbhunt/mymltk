import torch
import numpy as np
from mymltk.metrics import cross_entropy_binary, accuracy
from . import helpers

def sigmoid(z):
    """
    Sigmoid function
    """

    return 1 / (1 + torch.exp(-z))

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
        self.loss = None

        return
    
    def fit(self, X, y):
        """
        """
        X = helpers.to_tensor(X)
        y = helpers.to_tensor(y)
        m, n = X.shape
        weights = torch.rand(n, dtype=torch.float64) * 0.01
        bias = torch.zeros(1, dtype=torch.float64)
        self.loss = np.full(self.max_iter, np.nan)
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
            self.loss[i_iter] = cross_entropy_binary(h, y_)
            dw = 1 / b * (X_.T @ (h - y_))
            db = torch.mean(h - y_)
            weights -= (self.lr * dw)
            bias -= (self.lr * db)

        #
        self.weights = weights
        self.bias = bias

        return
    
    def predict(self, X):
        """
        """

        proba = self.predict_proba(X)

        return np.array([0 if proba[i] < 0.5 else 1 for i in range(len(proba))])
    
    def predict_proba(self, X):
        """
        """

        X_ = helpers.to_tensor(X)
        z = X_ @ self.weights + self.bias

        return sigmoid(z)
    
    def score(self, X, y, scoring="accuracy"):
        """
        """

        y_pred = self.predict(X)
        if scoring == "accuracy":
            score_ = accuracy(y_pred, y)
        else:
            raise Exception(f"{scoring} is not a supported scoring metric")

        return float(score_)