import datasets
import string
import numpy as np
from collections import Counter

class IMBDBDataset():
    """
    """

    def __init__(self, vocab_size=10000, sequence_length=500):
        """
        """

        #
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.imbd = None
        self._y_train = None
        self._y_test = None
        self._X_train = None
        self._X_test = None

        return
    
    def load(self):
        """
        """

        self.imbd = datasets.load_dataset("imdb")
        self._build_vocabulary()
        self._y_train = np.array([0 if l == 0 else 1 for l in self.imbd["train"]["label"]])
        self._y_test = np.array([0 if l == 0 else 1 for l in self.imbd["test"]["label"]])
        self._process_reviews()

        return
    
    def _build_vocabulary(self):
        """
        """

        counts = Counter()
        reviews = self.imbd["train"]["text"]
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
            "train": np.full([len(self.imbd["train"]["text"]), self.sequence_length], np.nan),
            "test": np.full([len(self.imbd["test"]["text"]), self.sequence_length], np.nan),
        }
        for split in ("train", "test"):
            reviews = self.imbd[split]["text"]
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