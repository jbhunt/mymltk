import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torchvision
from torchvision.transforms import v2
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from .helpers import is_tensor, to_tensor
from sklearn.datasets import load_digits
from torch.utils.data import DataLoader, TensorDataset

def _load_fashion():
    """
    """

    ds = torchvision.datasets.FashionMNIST(
        root="/home/josh/Downloads/data",
        transform=None,
        train=True
    )
    X = ds.data.reshape(-1, 1, 28, 28)
    mean = X.mean(dtype=torch.float)
    std = X.float().std()
    X = (X - mean) / std
    y = ds.targets

    return X, y

def load_digits():
    """
    """

    ds = load_digits()
    X = ds.data.reshape(-1, 1, 8, 8)
    y = ds.y

    return X, y

class _AlexNet(nn.Module):
    """
    """
    
    def __init__(self, C=1, n_classes=10, dropout=0.5):
        """
        """

        super().__init__()

        self.C = C

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=C, out_channels=96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(96),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )

        self.clf = nn.Sequential(  
            nn.Dropout(dropout),
            nn.LazyLinear(out_features=4096),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=n_classes)
        )

        return
    
    def forward(self, X):
        """
        """

        N = X.shape[0]
        x = self.cnn(X)
        x = torch.flatten(x, 1)
        logits = self.clf(x)

        return logits
    

class AlexNetClassifier():
    """
    """

    def __init__(self, batch_size=256, lr=0.001, dropout=0.5, max_iter=1000):
        """
        """

        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        self.max_iter = max_iter
        self.model = None
        self.loss = None
        self.encoder = None

        return
    
    def fit(self, X, y):
        """
        """

        #
        N, C, H, W = X.shape

        # Resize to 227 x 227
        X_ = to_tensor(X, dtype=torch.float)
        X_resized = F.interpolate(
            X_,
            size=227
        )

        #
        n_classes = len(np.unique(y))
        self.encoder = OneHotEncoder(
            categories=[np.arange(n_classes)],
            sparse_output=False
        )
        self.encoder.fit(y.reshape(-1, 1))
        y_indices = to_tensor(y, dtype=torch.float).long()

        #
        dataset = TensorDataset(X_resized, y_indices)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        #
        self.model = _AlexNet(C=C, n_classes=n_classes, dropout=self.dropout)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.loss = np.full(self.max_iter, np.nan)

        #
        self.model.train()
        for i_epoch in range(self.max_iter):

            #
            loss_total = 0
            n_batches = 0
            for X_batch, y_batch in loader:

                # Forward pass
                logits = self.model(X_batch)
                loss = loss_fn(logits, y_batch)
                loss_total += round(loss.item(), 6)
                n_batches += 1

                # Backpropagation
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            #
            self.loss[i_epoch] = loss_total / n_batches

        return
    
    def predict(self, X):
        """
        """

        if is_tensor(X) == False:
            X_ = to_tensor(X, dtype=torch.float)
        else:
            X_ = X
        n_samples = X_.shape[0]
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_)
            logits = logits.flatten(1)
            y_indices = torch.argmax(F.softmax(logits, dim=1), dim=1).detach().cpu().numpy()
        y_encoded = np.zeros([n_samples, len(self.encoder.categories[0])])
        y_encoded[np.arange(n_samples), y_indices] = 1
        y_pred = self.encoder.inverse_transform(y_encoded)

        return y_pred
    
    def score(self, X, y):
        """
        """

        return