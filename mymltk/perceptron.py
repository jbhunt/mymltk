import torch
import numpy as np
from mymltk.metrics import accuracy, cross_entropy_binary, missclassification_count

class PerceptronClassifier():
    """
    """

    def __init__(self, lr=0.001, max_iter=1000, shuffle=True):
        """
        """

        self.lr = lr
        self.weights = None
        self.max_iter = max_iter
        self.loss = None
        self.shuffle = shuffle  
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        return
    
    def fit(self, X, y):
        """
        """
    
        #
        if not (y.min() == -1 and y.max() == 1):
            raise Exception("Labels must be in the range -1 to +1")
        
        #
        if type(X) == np.ndarray:
            X = torch.from_numpy(X).to(self.device).double()
        if type(y) == np.ndarray:
            y = torch.from_numpy(y).to(self.device).double()

        #
        X_ = torch.hstack([X, torch.ones([X.shape[0], 1])])
        m, n = X_.shape
        weights = torch.rand(n, dtype=torch.double)
        i_iter = 0
        self.loss = np.full(self.max_iter, np.nan)
        while True:
            if i_iter >= self.max_iter:
                print("Warning: algorithm failed to converge")
                break
            sample_indices = np.arange(m)
            if self.shuffle:
                sample_indices = np.random.choice(sample_indices, replace=False)
            for i_sample in range(m):
                x_i = X_[i_sample]
                y_i = y[i_sample]
                y_prime = np.sign(weights.T @ x_i)
                if y_prime != y_i:
                    weights += self.lr * y_i * x_i.T
            y_pred = np.sign(X_ @ weights)
            loss = missclassification_count(y_pred, y)
            self.loss[i_iter] = loss
            if loss == 0:
                break
            i_iter += 1

        self.weights = weights

        return

    def predict(self, X):
        """
        """

        X_ = torch.hstack([torch.from_numpy(X), torch.ones([X.shape[0], 1])])
        return torch.sign(X_ @ self.weights)
    
    def score(self, X, y, scoring="accuracy"):
        """
        """

        y_pred = self.predict(X)
        if scoring == "accuracy":
            score_ = accuracy(y_pred, y)
        else:
            raise Exception(f"{scoring} is not a supported scoring metric")

        return float(score_)