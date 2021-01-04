import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from .BaseTokenizer import BaseTokenizer

class KerasTokenizer(BaseTokenizer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Tokenizer()
        self.max_length = None
        self.requires_save = True

    def name(self):
        return 'keras_tokenizer'

    def Tokenizer(self):
        return tf.keras.preprocessing.text.Tokenizer()

    def tokenize(self, x):
        self.tokenizer.fit_on_texts(x)
        return None

    def tokenize_and_pad(self, x, padding='post'):
        self.max_length = self._get_max_length(x)
        self.tokenize(x)
        sequences = self.tokenizer.texts_to_sequences(x)
        sequences = pad_sequences(sequences, maxlen=self.max_length, padding=padding)
        return sequences
