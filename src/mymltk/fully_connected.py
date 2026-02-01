import numpy as np
import torch
from torch import nn
from torch import optim
from sklearn.preprocessing import OneHotEncoder
from .helpers import to_tensor
from .metrics import accuracy

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
    return -1 * np.sum(y_true * np.log(y_pred), axis=1)

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
    
    def backward(self, y_true):
        """
        """

        # For scaling gradients to the batch size
        n_samples = len(y_true)

        # Partial derivatives for output layer
        dL_wrt_z_2 = self.y_hat - y_true
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
    
    def zero_grad(self):
        """
        """

        return
    
    def __call__(self, X):
        return self.forward(X)
    
class _FullyConnectedNeuralNetworkWithPyTorch(nn.Module):
    """
    """

    def __init__(self, input_size=1, hidden_size=1, output_size=1, device="cpu"):
        """
        """

        super().__init__()
        self.device = device
        self.l_hidden = nn.Linear(input_size, hidden_size, dtype=torch.float, device=device)
        self.relu_1 = nn.ReLU()
        self.l_out = nn.Linear(hidden_size, output_size, dtype=torch.float, device=device)
        # self.sm = nn.Softmax(dim=1)

        return

    def forward(self, X):
        """
        """

        X_ = to_tensor(X).float().to(self.device)
        z_1 = self.l_hidden(X_)
        a_1 = self.relu_1(z_1)
        z_2 = self.l_out(a_1)
        # y_hat = self.sm(z_2)

        return z_2

class FullyConnectedNeuralNetworkClassifier():
    """
    """

    def __init__(self, hidden_size=5, lr=0.001, max_iter=1000, backend="numpy", device=None):
        """
        """

        self.hidden_size = hidden_size
        self.lr = lr
        self.max_iter = max_iter
        self.model = None
        self.encoder = OneHotEncoder(sparse_output=False)
        self.loss = None
        self.backend = backend
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        return
    
    def _fit_with_numpy_backend(self, X, y):
        """
        """

        y_encoded = self.encoder.fit_transform(y.reshape(-1, 1))
        n_samples, n_features = X.shape
        self.model = _FullyConnectedNeuralNetworkWithNumPy(n_features, self.hidden_size, y_encoded.shape[1])
        self.loss = np.full(self.max_iter, np.nan)
        for i_epoch in range(self.max_iter):
            self.model.zero_grad()
            y_hat = self.model.forward(X)
            self.loss[i_epoch] = cross_entropy_loss(y_hat, y_encoded).mean()
            self.model.backward(y_encoded)

        return
    
    def _fit_with_pytorch_backend(self, X, y):
        """
        """

        X_ = torch.from_numpy(X).float().to(self.device)
        y_encoded = self.encoder.fit_transform(y.reshape(-1, 1))
        y_indices = torch.from_numpy(y_encoded.argmax(axis=1)).long().to(self.device)
        n_samples, n_features = X.shape
        self.model = _FullyConnectedNeuralNetworkWithPyTorch(
            input_size=n_features,
            hidden_size=self.hidden_size,
            output_size=y_encoded.shape[1],
            device=self.device
        )
        self.loss = np.full(self.max_iter, np.nan)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        for i_epoch in range(self.max_iter):
            logits = self.model(X_)
            loss = loss_fn(logits, y_indices)
            self.loss[i_epoch] = round(loss.item(), 6)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        return
    
    def fit(self, X, y):
        """
        """

        if self.backend in ["np", "numpy", "NumPy"]:
            self._fit_with_numpy_backend(X, y)
        elif self.backend in ["pytorch", "PyTorch", "torch"]:
            self._fit_with_pytorch_backend(X, y)

        return
    
    def predict(self, X):
        """
        """

        if self.backend in ["np", "numpy", "NumPy"]:
            y_hat = self.model(X)
        elif self.backend in ["pytorch", "PyTorch", "torch"]:
            with torch.no_grad():
                logits = self.model(X).numpy()
            y_hat = softmax(logits)
        y_pred = np.full_like(y_hat, 0)
        y_pred[np.arange(y_hat.shape[0]), np.argmax(y_hat, axis=1)] = 1
        y_pred = self.encoder.inverse_transform(y_pred.reshape(-1, 2)).flatten()

        return y_pred
    
    def score(self, X, y, scoring="accuracy"):
        """
        """

        y_pred = self.predict(X)
        y_true = y.flatten()
        if scoring == "accuracy":
            score_ = accuracy(y_pred, y_true)
        else:
            raise Exception(f"{scoring} is not a supported scoring metric")

        return float(score_)