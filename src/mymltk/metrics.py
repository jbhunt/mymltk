import torch
import numpy as np
from . import helpers

def _check_args(y_true, y_test):
    return (helpers.to_tensor(y_true), helpers.to_tensor(y_test))

def cross_entropy_binary(y_pred, y_true, epsilon=1e-15):
    """
    Binary cross-entropy loss
    """

    y_true, y_pred = _check_args(y_true, y_pred)
    y_pred = torch.clip(y_pred, epsilon, 1 - epsilon)
    loss = -1 * (torch.mean(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred)))

    return loss

def missclassification_count(y_pred, y_true):
    """
    Count of the number of misclassifications
    """

    y_true, y_pred = _check_args(y_true, y_pred)
    return (y_pred != y_true).sum()

def accuracy(y_pred, y_true):
    """
    Accuracy
    """

    y_true, y_pred = _check_args(y_true, y_pred)
    return torch.sum(y_pred == y_true) / len(y_true)

def mean_absolute_error(y_pred, y_true):
    """
    """
    y_true, y_pred = _check_args(y_true, y_pred)
    return torch.mean(torch.abs(y_pred - y_true))