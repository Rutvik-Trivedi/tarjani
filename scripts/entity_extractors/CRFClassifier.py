import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
import scipy

import warnings
warnings.filterwarnings('ignore')

from .BaseClassifier import BaseClassifier

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class CRFClassifier(BaseClassifier):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.requires_save = True


    def name(self):
        return 'crf_classifier'


    def modelling(self, **kwargs):
        crf = sklearn_crfsuite.CRF(
            algorithm=kwargs.get('algorithm', 'lbfgs'),
            max_iterations=kwargs.get('max_iterations', 100),
            all_possible_transitions=kwargs.get('all_possible_transitions', True)
            )
        default_param_space = {
        'c1': scipy.stats.expon(scale=0.5),
        'c2': scipy.stats.expon(scale=0.05),
        }
        f1_scorer = make_scorer(metrics.flat_f1_score,
                        average='weighted', labels=kwargs.get('labels'))
        rs = RandomizedSearchCV(crf, default_param_space,
                        cv=3,
                        verbose=1,
                        n_jobs=-1,
                        n_iter=50,
                        scoring=f1_scorer)
        return rs
