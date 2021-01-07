import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np

from .BaseFeaturizer import BaseFeaturizer

class EmbeddingFeaturizer(BaseFeaturizer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = max(len(kwargs.get('tokenizer').word_index)+1, 512)
        self.num_layers = kwargs.get('num_layers', 50)
        self.tokenizer = kwargs.get('tokenizer')
        self.embedding_matrix = self._create_embedding_matrix(
            embedding_file_path = kwargs.get('embedding_file_path'),
            num_layers=self.num_layers
        )
        self.requires_save = True

    def name(self):
        return 'embedding_featurizer'

    def modelling(self, **kwargs):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(kwargs.get('max_length', 20),)))
        model.add(tf.keras.layers.Embedding(self.vocab_size,
                            kwargs.get('num_layers',50),
                            weights = [kwargs['embedding_matrix']],
                            input_length = kwargs.get('max_length', 20),
                            trainable = kwargs.get('trainable_embeddings',False)))

        model.compile(optimizer = kwargs.get('optimizer', 'adam'),
                           loss = kwargs.get('loss', 'sparse_categorical_crossentropy'),
                           metrics = kwargs.get('metrics', ['accuracy']))

        return model


    def _create_embedding_matrix(self,
                                embedding_file_path,
                                num_layers):
        embedding_vector = {}
        f = open(embedding_file_path)
        for line in f:
            value = line.split(' ')
            word = value[0]
            coef = np.array(value[1:],dtype = 'float32')
            embedding_vector[word] = coef

        embedding_matrix = np.zeros((self.vocab_size,num_layers))
        for word,i in self.tokenizer.word_index.items():
                embedding_value = embedding_vector.get(word)
                if embedding_value is not None:
                    embedding_matrix[i] = embedding_value
        f.close()
        return embedding_matrix
