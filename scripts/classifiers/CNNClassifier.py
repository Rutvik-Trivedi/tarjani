import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

from .BaseClassifier import BaseClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, Flatten, Input, Conv1D, Embedding, MaxPooling1D
    )

class CNNClassifier(BaseClassifier):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.classifier_input_dim = kwargs.get('featurizer_output_dim')

    def name(self):
        return 'cnn_classifier'

    def get_kernel_size(self, maxlen):
        if maxlen==1 or maxlen==2:
            return 1
        else:
            return int(maxlen/3)

    def modelling(self, **kwargs):
        model = Sequential()
        model.add(Input(shape=self.classifier_input_dim))
        model.add(Conv1D(filters=kwargs.get('filters',32),
                         kernel_size=self.get_kernel_size(kwargs.get('maxlen',20)),
                         activation=kwargs.get('activation', 'relu')))
        model.add(MaxPooling1D(pool_size=kwargs.get('pool_size', 2)))
        model.add(Flatten())
        model.add(Dense(10, activation=kwargs.get('activation','relu')))
        model.add(Dense(kwargs.get('num_classes'),
                        activation=kwargs.get('classifier_activation', 'softmax')))

        return model

    def build(self, model, **kwargs):
        model.compile(
            loss=kwargs.get('loss','sparse_categorical_crossentropy'),
            optimizer=kwargs.get('optimizer','adam'),
            metrics=kwargs.get('metrics', ['accuracy']))
