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
        pass

    def __call__(self, **kwargs):
        self.entity_model = self.modelling(**kwargs)
        self.build(self.entity_model, **kwargs)

    def name(self):
        return 'base_classifier'

    def modelling(self, **kwargs):
        raise NotImplementedError

    def build(self, model, **kwargs):
        pass

    def train(self, train_X, train_y, **kwargs):
        return self.entity_model.fit(train_X, train_y)

    def predict(self, X):
        return self.entity_model.predict(X)

    def _model_type(self):
        return 'nlu'

    def _model_folder(self, intent):
        return '../../model/'+self._model_type()+'/'

    def save(self, name):
        if dill.pickles(self.entity_model):
            with open(name, 'wb') as f:
                pickle.dump(self.entity_model, f)
        else:
            self.entity_model.save(name)

    def load(self, name):
        try:
            with open(name, 'rb') as f:
                self.entity_model = pickle.load(f)
        except:
            self.entity_model = tf.keras.models.load_model(name)
