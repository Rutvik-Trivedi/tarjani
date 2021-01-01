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

    def name(self):
        return 'albert_featurizer'

    def modelling(self, **kwargs):
        input_ = tf.keras.layers.Input(shape=(), dtype=tf.string)
        albert = hub.KerasLayer(
            kwargs.get('model_path', '../../model/nlu/albert/albert_model'),
            kwargs.get('trainable', False)
            )
        tokenizer = hub.KerasLayer(
            kwargs.get('tokenizer_path', '../../model/nlu/albert/albert_tokenizer')
            )
        processed = tokenizer(input_)
        output = albert(processed)['pooled_output']
        model = tf.keras.models.Model(inputs=input_, outputs=output)
        return model


    @tf.function(experimental_relax_shapes=True)
    def predict(self, X):
        return self.model(X)
