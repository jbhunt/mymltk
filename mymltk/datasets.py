import datasets
import string
import numpy as np
from collections import Counter
import pathlib as pl
import h5py
from torch.utils.data import Dataset

class CustomDatasetMixin():
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

class IMBDBDataset(CustomDatasetMixin):
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
    
class BrainToText2025(Dataset, CustomDatasetMixin):
    """
    """

    def __init__(self, root=None, seq_len=int(0.02 * 1000 * 15), mode="power"):
        """
        """

        super().__init__()
        self.root = root
        self.seq_len = seq_len
        self.mode = mode
        self._z_train = None
        self._z_test = None
        self._seq_lens_train = None
        self._seq_lens_test = None
    
        return
    
    def load(self, n_sessions=None, padding_value=0):
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
        self._seq_lens_train = list()
        self._seq_lens_test = list()

        #
        if self.mode == "spikes":
            start_index = 0
            stop_index = 256
        elif self.mode == "power":
            start_index = 256
            stop_index = None

        #
        for session_index, folder in enumerate(pl.Path(self.root).iterdir()):
            if n_sessions is not None and session_index + 1 > n_sessions:
                break
            for file in folder.iterdir():
                if "test" in file.name:
                    continue
                X, y, seq_lens = list(), list(), list()
                with h5py.File(file, 'r') as stream:
                    for trial_index in stream.keys():
                        xi = np.array(stream[trial_index]["input_features"][:self.seq_len, start_index: stop_index]) #  T time bins x N channels
                        seq_len, n_features = xi.shape
                        seq_lens.append(seq_len)
                        if seq_len < self.seq_len:
                            n_elements = self.seq_len - seq_len
                            padding = np.full([n_elements, n_features], padding_value)
                            xi = np.vstack([xi, padding])
                        yi = stream[trial_index]["seq_class_ids"][:]
                        X.append(xi)
                        y.append(yi)
                X = np.array(X)
                n_trials, _, _ = X.shape
                z = np.tile(np.nanmean(X, axis=(0, 1)), n_trials).reshape(n_trials, -1)
                for xi, yi, zi, seq_len in zip(X, y, z, seq_lens):
                    if "train" in file.name:
                        self._X_train.append(xi)
                        self._y_train.append(yi)
                        self._z_train.append(zi)
                        self._seq_lens_train.append(seq_len)
                    elif "val" in file.name:
                        self._X_test.append(xi)
                        self._y_test.append(yi)
                        self._z_test.append(zi)    
                        self._seq_lens_test.append(seq_len)      

        #
        self._X_train = np.array(self._X_train)
        self._y_train = np.array(self._y_train)
        self._X_test = np.array(self._X_test)
        self._y_test = np.array(self._y_test)
        self._z_train = np.array(self._z_train)
        self._z_test = np.array(self._z_test)
        self._seq_lens_train = np.array(self._seq_lens_train)
        self._seq_lens_test = np.array(self._seq_lens_test)

        return
    
    @property
    def z_train(self):
        return self._z_train
    
    @property
    def z_test(self):
        return self._z_test
    
    @property
    def seq_lens_train(self):
        return self._seq_lens_train
    
    @property
    def seq_lens_test(self):
        return self._seq_lens_test
    
    def __len__(self):
        return self.X_train.shape[0]
    
    def __getitem__(self, index):
        return self.X_train[index], self.y_train[index], self.z_train[index], self.seq_lens_train[index]