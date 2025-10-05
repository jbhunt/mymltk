import numpy as np

class LinearRegressionWithGD():
    """
    """

    def __init__(self, max_iter=1000, lr=1e-3):
        self.max_iter = max_iter
        self.lr = lr
        self.weights = None
        self.bias = None
        self.loss = None
        return
    
    def fit(self, X, y):
        """
        """

        X_with_bias = np.concatenate([
            np.ones(X.shape[0]).reshape(-1, 1),
            X,
        ], axis=1)
        weights = np.zeros(X_with_bias.shape[1])
        n = X.shape[0]
        self.loss = np.full(self.max_iter, np.nan)

        for i_step in range(self.max_iter):

            # Compute the gradient
            residuals = np.matmul(X_with_bias, weights) - y
            gradient = np.matmul(X_with_bias.T, residuals) / n

            # Update the weights
            weights = weights - self.lr * gradient

            # Record loss
            mse = np.mean(np.power(residuals, 2))
            self.loss[i_step] = mse

        #
        self.bias = weights[0]
        self.weights = weights[1:]

        return
    
    def predict(self, X):
        """
        """

        return X @ self.weights + self.bias
    
    def score(self, X, y, metric="r2"):
        """
        """

        y_pred = self.predict(X)
        if metric == "mse":
            score = np.mean(np.power(y_pred - y, 2))
        elif metric == "rmse":
            score = np.sqrt(np.mean(np.power(y_pred - y, 2)))
        elif metric == "r2":
            score = np.sum(np.power(y - y_pred, 2)) / np.sum(np.power(y - y.mean(), 2))
        else:
            raise Exception(f"{metric} is not a supported metric")
        score = round(score, 3)

        return score