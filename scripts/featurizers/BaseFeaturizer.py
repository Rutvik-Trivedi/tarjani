import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import dill
import pickle


import tensorflow as tf
import tensorflow_text
class BaseFeaturizer():
    '''
    Similar to the torch.nn Module class for featurizers.
    You will need to inherit this class while creating a new
    model architecture.
    '''

    def __init__(self, **kwargs):
        self.featurizer_model = None
        self.featurizer_output_dim = None

    def __call__(self, **kwargs):
        self.featurizer_model = self.modelling(**kwargs)
        self.featurizer_output_dim = self._get_featurizer_output_dim()

    def _get_featurizer_output_dim(self):
        dim = self.featurizer_model.layers[-1].output.shape
        if dim[0] is None:
            return dim[1:]
        return dim

    def name(self):
        return 'base_featurizer'

    def modelling(self, **kwargs):
        raise NotImplementedError

    def predict(self, X):
        return self.featurizer_model(X)

    def save(self, name):
        if dill.pickles(self.featurizer_model):
            with open(name, 'wb') as f:
                pickle.dump(self.featurizer_model, f)
        else:
            self.featurizer_model.save(name)

    def load(self, name):
        try:
            with open(name, 'rb') as f:
                self.featurizer_model = pickle.load(f)
        except:
            self.featurizer_model = tf.keras.models.load_model(name)
