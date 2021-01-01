import sklearn_crfsuite

from .BaseClassifier import BaseClassifier

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class CRFClassifier(BaseClassifier):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def name(self):
        return 'crf_classifier'


    def modelling(self, **kwargs):
        crf = sklearn_crfsuite.CRF(
            algorithm=kwargs.get('algorithm', 'lbfgs'),
            c1=kwargs.get('c1', 0.6),
            c2=kwargs.get('c2', 0.01),
            max_iterations=kwargs.get('max_iterations', 100),
            all_possible_transitions=kwargs.get('all_possible_transitions', True)
            )
        return crf


    def _model_folder(self, intent):
        return '../../intents/'+intent+'/'

    def save(self, intent=None):
        with open(self._model_folder(intent)+'entity.tarjani', 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, intent):
        with open(self._model_folder(intent)+'entity.tarjani', 'wb') as f:
            self.model = pickle.load(f)
