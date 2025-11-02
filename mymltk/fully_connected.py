import numpy as np
from torch import nn
from sklearn.preprocessing import OneHotEncoder

def relu(x):
    """
    Rectified linear unit function (ReLU)
    """

    return np.maximum(x, 0)

def relu_prime(x):
    """
    Derivative of the ReLU function
    """

    dx = np.zeros_like(x)
    dx[x > 0] = 1
    return dx

def softmax(x):
    """
    Softmax function
    """

    # Subtract max for numerical stability
    exponentiated = np.exp(x - x.max(axis=1).reshape(-1, 1)) 
    return exponentiated / np.sum(exponentiated, axis=1).reshape(-1, 1)

def cross_entropy_loss(y_pred, y_true):
    """
    """

    y_true = np.clip(y_true, 1e-12, np.inf)
    return -1 * np.sum(y_pred * np.log(y_true), axis=1)

class _FullyConnectedNeuralNetworkWithNumPy():
    """
    """

    def __init__(self, input_size=1, hidden_size=1, output_size=1, lr=0.001):
        """
        """

        #
        self.lr = lr
        self.W_hidden = np.random.randn(input_size, hidden_size) * 0.2 # Weights connecting inputs to hidden layer
        self.b_hidden = np.zeros((1, hidden_size)) # Bias for hidden layer
        self.W_out = np.random.randn(hidden_size, output_size) * 0.1 # Weights connecting hidden layer to output layer
        self.b_out = np.zeros((1, output_size)) # Bias for output layer

        #
        self.X = None
        self.z_1 = None
        self.a_1 = None
        self.z_2 = None
        self.a_2 = None
        self.y_hat = None

        return

    def forward(self, X):
        """
        """

        self.X = X
        self.z_1 = X @ self.W_hidden + self.b_hidden
        self.a_1 = relu(self.z_1)
        self.z_2 = self.a_1 @ self.W_out + self.b_out
        self.a_2 = softmax(self.z_2)
        self.y_hat = self.a_2

        return self.a_2
    
    def backward(self, y):
        """
        """

        # For scaling gradients to the batch size
        n_samples = len(y)

        # Partial derivatives for output layer
        dL_wrt_z_2 = self.y_hat - y
        dL_wrt_W_out = self.a_1.T @ dL_wrt_z_2 / n_samples
        dL_wrt_b_out = np.sum(dL_wrt_z_2, axis=0, keepdims=True) / n_samples

        # Partial derivatives for hidden layer
        dL_wrt_a_1 = dL_wrt_z_2 @ self.W_out.T
        dL_wrt_z_1 = dL_wrt_a_1 * relu_prime(self.z_1)
        dL_wrt_W_hidden = self.X.T @ dL_wrt_z_1 / n_samples
        dL_wrt_b_hidden = np.sum(dL_wrt_z_1, axis=0, keepdims=True) / n_samples

        # Update parameters
        self.W_hidden += -1 * self.lr * dL_wrt_W_hidden
        self.b_hidden += -1 * self.lr * dL_wrt_b_hidden
        self.W_out += -1 * self.lr * dL_wrt_W_out
        self.b_out += -1 * self.lr * dL_wrt_b_out

        return
    
# TODO: Implement FC network with PyTorch
class _FullyConnectedNeuralNetworkWithPyTorch(nn.Module):
    """
    """

class FullyConnectedNeuralNetworkClassifier():
    """
    """

    def __init__(self, hidden_size=5, lr=0.001, max_iter=1000):
        """
        """

        self.hidden_size = hidden_size
        self.lr = lr
        self.max_iter = max_iter
        self.model = None
        self.encoder = OneHotEncoder(sparse_output=False)
        self.loss = None

        return
    
    def fit(self, X, y):
        """
        """

        y_encoded = self.encoder.fit_transform(y.reshape(-1, 1))
        n_samples, n_features = X.shape
        self.model = _FullyConnectedNeuralNetworkWithNumPy(n_features, self.hidden_size, y_encoded.shape[1])
        self.loss = np.full(self.max_iter, np.nan)
        for i_epoch in range(self.max_iter):
            a_2 = self.model.forward(X)
            self.loss[i_epoch] = cross_entropy_loss(a_2, y_encoded).mean()
            self.model.backward(y_encoded)

        return
    
    def predict(self, X):
        """
        """

        y_encoded = self.model.forward(X)
        y_pred = self.encoder.inverse_transform(y_encoded)
        return y_pred
    
    def score(self, X, y):
        """
        """

        return