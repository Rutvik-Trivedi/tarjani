import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import dill
import pickle
import tensorflow as tf

class BaseTokenizer():
    '''
    The base class for all tokenizers. Need to inherit this class
    for creating your own tokenizers
    '''

    def __init__(self, **kwargs):
        self.tokenizer = None

    def __call__(self):
        self.tokenizer = self.Tokenizer()

    def name(self):
        return 'base_tokenizer'

    def _get_max_length(self, x):
        max_length = -1
        for i in x:
            if len(i.split(' ')) > max_length:
                max_length = len(i.split(' '))
        return max_length

    def Tokenizer(self):
        raise NotImplementedError

    def tokenize(self, x):
        raise NotImplementedError

    def tokenize_and_pad(self, x, padding='post'):
        raise NotImplementedError

    def save(self, name):
        if dill.pickles(self.tokenizer):
            with open(name, 'wb') as f:
                pickle.dump(self.tokenizer, f)
        else:
            self.tokenizer.save(name)

    def load(self, name):
        try:
            with open(name, 'rb') as f:
                self.tokenizer = pickle.load(f)
        except:
            self.tokenizer = tf.keras.models.load_model(name)
