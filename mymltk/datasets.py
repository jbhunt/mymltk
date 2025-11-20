import datasets
import string
import numpy as np
from collections import Counter
import pathlib as pl
import h5py

class CustomDataset():
    """
    """

    def __init__(self):
        """
        """

        self._data = None
        self._y_train = None
        self._y_test = None
        self._X_train = None
        self._X_test = None

        return
    
    @property
    def X_train(self):
        return self._X_train
    
    @property
    def X_test(self):
        return self._X_test
    
    @property
    def y_train(self):
        return self._y_train
    
    @property
    def y_test(self):
        return self._y_test

class IMBDBDataset(CustomDataset):
    """
    """

    def __init__(self, vocab_size=10000, sequence_length=500):
        """
        """

        #
        super().__init__()

        #
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self._data = None

        return
    
    def load(self):
        """
        """

        self._data = datasets.load_dataset("imdb")
        self._build_vocabulary()
        self._y_train = np.array([0 if l == 0 else 1 for l in self._data["train"]["label"]])
        self._y_test = np.array([0 if l == 0 else 1 for l in self._data["test"]["label"]])
        self._process_reviews()

        return
    
    def _build_vocabulary(self):
        """
        """

        counts = Counter()
        reviews = self._data["train"]["text"]
        for review in reviews:
            translator = str.maketrans("", "", string.punctuation)
            cleaned = review.lower().translate(translator)
            words = cleaned.split(" ")
            counts.update(words)
        most_common_words = ["<padding>", "<unknown>"] + [word for word, count in counts.most_common(self.vocab_size - 2)]
        self.vocab = {word: i for i, word in enumerate(most_common_words)}

        return
    
    def _process_reviews(self):
        """
        """

        X = {
            "train": np.full([len(self._data["train"]["text"]), self.sequence_length], np.nan),
            "test": np.full([len(self._data["test"]["text"]), self.sequence_length], np.nan),
        }
        for split in ("train", "test"):
            reviews = self._data[split]["text"]
            for i_review, review in enumerate(reviews):
                translator = str.maketrans("", "", string.punctuation)
                cleaned = review.lower().translate(translator)
                words = cleaned.split(" ")
                words_encoded = np.array([self.vocab[word] if word in self.vocab.keys() else self.vocab["<unknown>"] for word in words])
                if words_encoded.size < self.sequence_length:
                    n = self.sequence_length - words_encoded.size
                    words_encoded = np.concatenate([
                        words_encoded,
                        np.full(n, self.vocab["<padding>"])
                    ])
                elif words_encoded.size > self.sequence_length:
                    words_encoded = words_encoded[:self.sequence_length]
                X[split][i_review] = words_encoded

        #
        self._X_train = X["train"]
        self._X_test = X["test"]

        return
    
class BrainToText2025(CustomDataset):
    """
    """

    def __init__(self, root=None, max_seq_len=int(0.02 * 1000 * 15)):
        """
        """

        super().__init__()
        self.root = root
        self.max_seq_len = max_seq_len
        self._z_train = None
        self._z_test = None
    
        return
    
    def load(self):
        """
        """

        if self.root is None:
            raise Exception("Root directory not sepecified during instantiation")

        self._X_train = list()
        self._y_train = list()
        self._X_test = list()
        self._y_test = list()
        self._z_train = list()
        self._z_test = list()

        #
        for session_index, folder in enumerate(pl.Path(self.root).iterdir()):
            for file in folder.iterdir():
                if "test" in file.name:
                    continue
                X, y, z = list(), list(), list()
                with h5py.File(file, 'r') as stream:
                    for trial_index in stream.keys():
                        xi = stream[trial_index]["input_features"][:self.max_seq_len, :] # T time bins x N features
                        seq_len, n_features = xi.shape
                        xi_flat = xi.flatten() # Size of T * N
                        if seq_len < self.max_seq_len:
                            n_elements = (self.max_seq_len * n_features) - len(xi_flat)
                            xi_flat = np.pad(xi_flat, [0, n_elements], constant_values=np.nan)
                        X.append(xi_flat)
                        yi = stream[trial_index]["seq_class_ids"][:]
                        y.append(yi)
                        z.append(session_index)
                for xi, yi, zi in zip(X, y, z):
                    if "train" in file.name:
                        self._X_train.append(xi)
                        self._y_train.append(yi)
                        self._z_train.append(zi)
                    elif "val" in file.name:
                        self._X_test.append(xi)
                        self._y_test.append(yi)
                        self._z_test.append(zi)

        #
        self._X_train = np.array(self._X_train)
        self._y_train = np.array(self._y_train)
        self._X_test = np.array(self._X_test)
        self._y_test = np.array(self._y_test)
        self._z_train = np.array(self._z_train)
        self._z_test = np.array(self._z_test)

        return
    
    @property
    def z_train(self):
        return self._z_train
    
    @property
    def z_test(self):
        return self._z_test