import numpy as np
from sklearn.metrics import confusion_matrix

class KNearestNeighborClassifier():
    def __init__(
        self,
        k=3,
        norm=2
        ):
        """
        """

        self.k = k
        self.norm = norm
        self.X = None
        self.y = None

        return
    
    def fit(self, X, y):
        """
        Store the samples and labels
        """
        self.X = X
        self.y = y


        return
    
    def predict(self, X):
        """
        Predict the label of each sample from X
        """

        y_pred = list()
        for xi in X:
            d = np.power(np.sum(np.power(np.abs(self.X - xi), self.norm), axis=1), 1 / self.norm)
            y_indices = np.argsort(d)[:self.k]
            y_closest = self.y[y_indices]
            unique_values, counts = np.unique(y_closest, return_counts=True)
            i_value = np.argmax(counts)
            y_pred.append(
                unique_values[i_value]
            )

        return np.array(y_pred)
    
    def score(self, X, y, metric="accuracy"):
        """
        Measure the performance of the classifier
        """

        y_pred = self.predict(X)
        if metric == "accuracy":
            score_ = np.sum(y_pred == y) / y_pred.size
        else:
            raise Exception(f"{metric} is not yet implemented")

        return score_