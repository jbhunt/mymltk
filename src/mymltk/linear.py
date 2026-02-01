import torch
from torch import nn
import numpy as np
from . import helpers, metrics

class _LinearModel(nn.Module):
    """
    """

    def __init__(self, d_in=1, d_out=1):
        """
        """

        super().__init__()
        self.linear = nn.Linear(d_in, d_out, bias=True, dtype=torch.double)

        return
    
    def forward(self, x):
        """
        """
        z = self.linear(x)
        return z
    
class LinearRegression():
    """
    """

    def __init__(self, lr=0.001, max_iter=1000):
        """
        """
        self.lr = lr
        self.max_iter = max_iter
        self.model = None
        self.loss = None
        return
    
    def fit(self, X, y):
        """
        """

        #
        X_ = helpers.to_tensor(X)
        y_ = np.atleast_2d(y).reshape(-1, 1)
        y_ = helpers.to_tensor(y_)

        #
        d_in = X.shape[1]
        d_out = y_.shape[1]
        self.model = _LinearModel(d_in, d_out)
        loss_fn = nn.MSELoss()
        self.loss = np.full(self.max_iter, np.nan)

        #
        for i_epoch in range(self.max_iter):

            # Zero the gradients
            for p in self.model.parameters():
                p.grad = None
            
            # Forward pass
            z = self.model(X_)
            loss = loss_fn(z, y_)
            self.loss[i_epoch] = loss.detach().numpy()

            # Compute gradients
            loss.backward()
            grad_weights = self.model.linear.weight.grad
            grad_bias = self.model.linear.bias.grad

            # Update weights and bias
            with torch.no_grad():
                self.model.linear.weight.add_(-1 * self.lr * grad_weights)
                self.model.linear.bias.add_(-1 * self.lr * grad_bias)

        return
    
    def predict(self, X):
        """
        """
        X_ = helpers.to_tensor(X)
        z = self.model(X_)
        return z.detach().numpy()
    
    def score(self, X, y, metric="mae"):
        y_pred = self.predict(X).ravel()
        y_ = y.ravel()
        if metric == "mae":
            score = float(metrics.mean_absolute_error(y_pred, y_))
        else:
            raise Exception(f"{metric} is not a supported scoring metric")

        return score