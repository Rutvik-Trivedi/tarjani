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
        self.classifier_input_dim = None

    def __call__(self, **kwargs):
        self.classifier_input_dim = kwargs.get('featurizer_output_dim')
        self.model = self.modelling(**kwargs)
        self.build(self.model, **kwargs)

    def name(self):
        return 'base_classifier'


    def modelling(self, **kwargs):
        raise NotImplementedError

    def build(self, model, **kwargs):
        pass

    def train(self, train_X, train_y, **kwargs):
        history = self.model.fit(train_X, train_y,
                                 kwargs.get('verbose',1), kwargs.get('epochs',20))
        return history

    def predict(self, X):
        return self.model.predict(X)

    def _model_type(self):
        return 'nlu'

    def _model_folder(self):
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
