from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib import pyplot as plt
from .logistic import LogisticRegressionClassifier
from .perceptron import RosenblattPerceptronClassifier
from .fully_connected import FullyConnectedNeuralNetworkClassifier
from .helpers import rescale
import numpy as np

def visualize_classifier_performance(k=10):
    """
    """

    #
    ds = load_breast_cancer()
    X = ds.data
    y = ds.target

    #
    clfs = {
        "logistic_regression": LogisticRegressionClassifier(max_iter=10000, lr=0.001),
        "perceptron": RosenblattPerceptronClassifier(max_iter=1000, lr=0.001),
        "neural_network": FullyConnectedNeuralNetworkClassifier(max_iter=10000, lr=0.001)
    }
    cms = {
        "logistic_regression": np.full([k, 2, 2], np.nan),
        "perceptron": np.full([k, 2, 2], np.nan),
        "neural_network": np.full([k, 2, 2], np.nan)
    }
    accuracy = {
        "logistic_regression": np.full(k, np.nan),
        "perceptron": np.full(k, np.nan),
        "neural_network": np.full(k, np.nan)
    }
    cv = StratifiedShuffleSplit(n_splits=k, train_size=0.8)
    for k_ in clfs.keys():
        clf = clfs[k_]
        for i_split, (i_train, i_test) in enumerate(cv.split(X, y)):
            X_train = X[i_train, :]
            y_train = y[i_train]
            X_test = X[i_test, :]
            y_test = y[i_test]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy[k_][i_split] = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred).astype(float)
            cm_normed = cm.copy()
            cm_normed[0, :] = cm[0, :] / cm[0, :].sum()
            cm_normed[1, :] = cm[1, :] / cm[1, :].sum()
            cms[k_][i_split] = cm_normed

    #
    fig, axs = plt.subplots(ncols=3, sharex=True, sharey=True)
    titles = {
        "logistic_regression": "Logistic regression",
        "perceptron": "Rosenblatt perceptron",
        "neural_network": "Neural network"
    }
    fontcolors = np.array([
        ['k', 'w'],
        ['w', 'k']
    ])
    for k_, clf, ax in zip(clfs.keys(), clfs.items(), axs):
        z_mean = cms[k_].mean(0)
        z_ci = 1.96 * cms[k_].std(0) / np.sqrt(k) # 95% CI for the mean
        im = ax.imshow(z_mean, vmin=0, vmax=1)
        for (i, j), z_mean_ij in np.ndenumerate(z_mean):
            z_ci_ij = z_ci[i, j]
            ci_lower_bound = np.clip(z_mean_ij - z_ci_ij, 0, 1)
            ci_upper_bound = np.clip(z_mean_ij + z_ci_ij, 0, 1)
            text = f"{z_mean_ij:.2f}\n[{ci_lower_bound:.2f}-{ci_upper_bound:.2f}]"
            ax.text(j, i, text, ha="center", va="center", color=str(fontcolors[i, j]))
        ax.set_xticks([0, 1])
        ax.set_yticks([1, 0])
        ax.set_xlabel("Predicted labels")
        ax.set_title(titles[k_])
    axs[0].set_ylabel("True labels")
    cb = fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.05, pad=0.05,label="Frac. of true labels", shrink=0.8, aspect=10)
    fig.set_figwidth(10)
    fig.set_figheight(4)
    fig.show()

    return fig, axs