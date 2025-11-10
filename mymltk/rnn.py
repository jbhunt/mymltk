from .datasets import IMBDBDataset
from .helpers import is_tensor, to_tensor
from torch.nn import Module
from torch import nn, optim
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from torch.nn import functional as F

class TextSequenceRNN(Module):
    """
    """

    def __init__(self, vocab_size=10000, embedding_size=100, hidden_size=128):
        """
        """

        super().__init__()
        self.l_embed = nn.Embedding(vocab_size, embedding_size)
        self.l_rnn = nn.LSTM(embedding_size, hidden_size, num_layers=1, batch_first=True)
        self.l_out = nn.Linear(hidden_size, 2)

        return
    
    def forward(self, X):
        """
        """

        embedded = self.l_embed(X)
        output, (h_n, c_n) = self.l_rnn(embedded)
        h_last = h_n[-1]
        logits = self.l_out(h_last)

        return logits
    

class TextSequenceSentimentClassifier():
    """
    """

    def __init__(self, batch_size=128, max_iter=100, embedding_size=100, hidden_size=128, lr=0.001):
        """
        """
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.loss = None
        return

    def fit(self, X, y):
        """
        """

        L = X.shape[1]
        V = int(X.max() + 1)
        X_ = to_tensor(X, dtype=torch.long)
        y_ = to_tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_, y_)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model = TextSequenceRNN(vocab_size=V, embedding_size=self.embedding_size, hidden_size=self.hidden_size)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss = np.full(self.max_iter, np.nan)

        #
        self.model.train()
        for i_epoch in range(self.max_iter):

            #
            self.loss[i_epoch] = 0.0

            # Learn over mini-batches
            for i_batch, (X_batch, y_batch) in enumerate(loader):

                #
                self.model.zero_grad()

                if i_batch >= 100:
                    break
                
                # Forward pass
                logits = self.model(X_batch)

                # Compute loss
                loss = loss_fn(logits, y_batch)
                self.loss[i_epoch] += loss.item()

                # Backprop and adjust weights
                loss.backward()
                optimizer.step()

            #
            self.loss[i_epoch] /= (i_batch + 1)

        return
    
    def predict(self, X):
        """
        """

        if is_tensor(X) == False:
            X_ = to_tensor(X, dtype=torch.long)
        else:
            X_ = X
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_)
            y_indices = torch.argmax(F.softmax(logits, dim=1), dim=1).detach().cpu().numpy()

        return y_indices