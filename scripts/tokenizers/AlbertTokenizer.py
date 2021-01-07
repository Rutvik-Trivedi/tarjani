import logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s]:%(asctime)s:%(message)s")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub

from .BaseTokenizer import BaseTokenizer

class AlbertTokenizer(BaseTokenizer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.requires_save = False

    def name(self):
        return 'albert_tokenizer'

    def __call__(self, **kwargs):
        self.tokenizer_output_dim = (128,)
        self.tokenizer = self.Tokenizer()

    def Tokenizer(self, **kwargs):
        input_ = tf.keras.layers.Input(shape=(), dtype=tf.string)
        try:
            tokenizer = hub.KerasLayer(
                kwargs.get('tokenizer_path', '../model/nlu/albert/albert_tokenizer')
                )
        except hub.resolver.UnsupportedHandleError:
            logging.error("This pipeline requires Albert Model to be downloaded. Please visit https://github.com/Rutvik-Trivedi/tarjani-model-zoo and follow the instructions in the documentation")
            exit()
        processed = tokenizer(input_)
        model = tf.keras.models.Model(inputs=input_, outputs=processed)
        return model

    def tokenize(self, x):
        x = tf.cast(x, dtype=tf.string)
        return self.tokenizer(x)

    @tf.function(experimental_relax_shapes=True)
    def tokenize_and_pad(self, x):
        return self.tokenize(x)

    @tf.function(experimental_relax_shapes=True)
    def prepare_query(self, x, **kwargs):
        return self.tokenize(x)
