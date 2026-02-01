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
    
class PhonemeVocabulary():
    """
    """

    PAD = 0
    BOS = 1
    EOS = 2
    map = {
        0: "<PAD>",
        1: "<BOS>",
        2: "<EOS>",
        3: "<BLANK>",
        4: "AA",
        5: "AE",
        6: "AH",
        7: "AO",
        8: "AW",
        9: 'AY',
        10: 'B',
        11: 'CH',
        12: 'D',
        13: 'DH',
        14: 'EH',
        15: 'ER',
        16: 'EY',
        17: 'F',
        18: 'G',
        19: 'HH',
        20: 'IH', 
        21: 'IY',
        22: 'JH',
        23: 'K',
        24: 'L',
        25: 'M',
        26: 'N',
        27: 'NG',
        28: 'OW',
        29: 'OY',
        30: 'P',
        31: 'R',
        32: 'S',
        33: 'SH',
        34: 'T',
        35: 'TH',
        36: 'UH',
        37: 'UW',
        38: 'V',
        39: 'W',
        40: 'Y',
        41: 'Z',
        42: 'ZH',
        43: '<SILENCE>'
    }

    def __init__(self):
        """
        """

        return
    
    def insert_special_tokens(self, in_seq, padding_token=0):
        """
        """

        in_seq = np.array(in_seq)
        out_seq = np.copy(in_seq)
        mask = np.array(out_seq) != padding_token
        out_seq[mask] += 2
        i = np.where(in_seq ==  0)[0][0]
        out_seq = np.insert(out_seq, i, self.EOS)
        out_seq = np.concatenate([
            np.array([self.BOS]),
            out_seq
        ])

        return out_seq
    
    def decode(self, in_seq):
        """
        """

        out_seq = list()
        for el in in_seq:
            token = self.map[el]
            out_seq.append(token)

        return np.array(out_seq)
    
    @property
    def size(self):
        return max(self.map.keys()) + 1
    
class BrainToText2025(Dataset, CustomDatasetMixin):
    """
    """

    def __init__(self, root=None, src_seq_len=int(0.02 * 1000 * 15), mode="power"):
        """
        """

        super().__init__()
        self.root = root
        self.src_seq_len = src_seq_len
        self.mode = mode
        self._z_train = None
        self._z_test = None
        self._src_seq_lens_train = None
        self._src_seq_lens_test = None
        self.vocab = PhonemeVocabulary()
    
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
        self._src_seq_lens_train = list()
        self._src_seq_lens_test = list()

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
                is_test_data = False
                if "test" in file.name:
                    is_test_data = True
                X, y, src_seq_lens, tgt_seq_lens = list(), list(), list(), list()
                with h5py.File(file, 'r') as stream:
                    for trial_index in stream.keys():
                        xi = np.array(stream[trial_index]["input_features"][:self.src_seq_len, start_index: stop_index]) #  T time bins x N channels
                        src_seq_len, n_features = xi.shape
                        src_seq_lens.append(src_seq_len)
                        if src_seq_len < self.src_seq_len:
                            n_elements = self.src_seq_len - src_seq_len
                            padding = np.full([n_elements, n_features], padding_value)
                            xi = np.vstack([xi, padding])

                        if is_test_data == False:
                            yi = stream[trial_index]["seq_class_ids"][:]
                            yi = self.vocab.insert_special_tokens(yi)
                        else:
                            yi = np.nan
                        y.append(yi)
                        X.append(xi)
                X = np.array(X)
                n_trials, _, _ = X.shape
                z = np.tile(np.nanmean(X, axis=(0, 1)), n_trials).reshape(n_trials, -1)
                for xi, yi, zi, src_seq_len in zip(X, y, z, src_seq_lens):
                    if "train" in file.name or "val" in file.name:
                        self._X_train.append(xi)
                        self._y_train.append(yi)
                        self._z_train.append(zi)
                        self._src_seq_lens_train.append(src_seq_len)
                    elif "test" in file.name:
                        self._X_test.append(xi)
                        self._z_test.append(zi)    
                        self._src_seq_lens_test.append(src_seq_len)      

        #
        self._X_train = np.array(self._X_train)
        self._y_train = np.array(self._y_train)
        self._X_test = np.array(self._X_test)
        self._y_test = None
        self._z_train = np.array(self._z_train)
        self._z_test = np.array(self._z_test)
        self._seq_lens_train = np.array(self._src_seq_lens_train)
        self._seq_lens_test = np.array(self._src_seq_lens_test)

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