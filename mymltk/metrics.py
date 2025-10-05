import numpy as np

def bce(y_pred, y_true, epsilon=1e-15):
    """
    Binary cross-entropy loss
    """

    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    return loss

def accuracy(y_pred, y_true):
    """
    """

    return np.sum(y_pred == y_true) / y_true.size
