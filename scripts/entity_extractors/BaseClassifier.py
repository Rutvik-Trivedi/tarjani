import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import dill
import pickle
import tensorflow as tf

class BaseClassifier():
    '''
    Similar to the torch.nn Module class.
    You will need to inherit this class while creating a new
    model architecture.
    '''

    def __init__(self, **kwargs):
        self.model = None

    def name(self):
        return 'base_classifier'

    def modelling(self, **kwargs):
        raise NotImplementedError

    def build(self, model, **kwargs):
        pass

    def train(self, train_X, train_y, **kwargs):
        self.model = self.modelling(**kwargs)
        self.build(self.model, **kwargs)
        history = self.model.fit(train_X, train_y, **kwargs)
        return history

    def predict(self, X):
        return self.model.predict(X)

    def _model_type(self):
        return 'nlu'

    def _model_folder(self, intent):
        return '../../model/'+self._model_type()+'/'

    def save(self, name):
        if dill.pickles(self.model):
            with open(name, 'wb') as f:
                pickle.dump(self.model, f)
        else:
            self.model.save(name)

    def load(self, name):
        try:
            with open(name, 'rb') as f:
                self.model = pickle.load(f)
        except:
            self.model = tf.keras.models.load_model(name)