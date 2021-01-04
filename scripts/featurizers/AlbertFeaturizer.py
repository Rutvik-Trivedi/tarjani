import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub

from .BaseFeaturizer import BaseFeaturizer

class AlbertFeaturizer(BaseFeaturizer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.featurizer_output_dim = (768,)
        self.requires_save = False

    def name(self):
        return 'albert_featurizer'

    def modelling(self, **kwargs):
        input_word_ids = tf.keras.layers.Input(
            shape=kwargs.get('tokenizer_output_dim'),
            name='input_word_ids',
            dtype=tf.int32,
            )
        input_mask = tf.keras.layers.Input(
            shape=kwargs.get('tokenizer_output_dim'),
            name='input_mask',
            dtype=tf.int32,
            )
        input_type_ids = tf.keras.layers.Input(
            shape=kwargs.get('tokenizer_output_dim'),
            name='input_type_ids',
            dtype=tf.int32,
            )
        albert = hub.KerasLayer(
            kwargs.get('model_path', '../model/nlu/albert/albert_model'),
            kwargs.get('trainable', False)
            )
        output = albert(
            {
                'input_word_ids': input_word_ids,
                'input_mask': input_mask,
                'input_type_ids': input_type_ids
            }
            )['pooled_output']

        model = tf.keras.models.Model(
            inputs={
                'input_word_ids': input_word_ids,
                'input_mask': input_mask,
                'input_type_ids': input_type_ids
            },
            outputs=output
            )

        return model


    @tf.function(experimental_relax_shapes=True)
    def predict(self, X):
        return self.featurizer_model(X)
