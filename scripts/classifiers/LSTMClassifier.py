import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

from .BaseClassifier import BaseClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, LSTM, Embedding

class LSTMClassifier(BaseClassifier):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.classifier_input_dim = kwargs.get('featurizer_output_dim')
        self.requires_save = True


    def name(self):
        return 'lstm_classifier'


    def modelling(self, **kwargs):
        model = Sequential()
        model.add(Input(shape=self.classifier_input_dim))
        model.add(LSTM(kwargs.get('num_layers',50)))
        model.add(Dropout(kwargs.get('dropout', 0.2)))
        model.add(Flatten())
        model.add(Dense(kwargs.get('num_classes'),
                        activation=kwargs.get('activation', 'softmax')))
        return model

    def build(self, model, **kwargs):
        model.compile(optimizer = kwargs.get('optimizer', 'adam'),
                      loss = kwargs.get('loss', 'sparse_categorical_crossentropy'),
                      metrics = kwargs.get('metrics', ['accuracy']))
