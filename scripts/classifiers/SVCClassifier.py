from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from .BaseClassifier import BaseClassifier


class SVCClassifier(BaseClassifier):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.classifier_input_dim = kwargs.get('featurizer_output_dim', None)


    def name(self):
        return 'svm_classifier'


    def modelling(self, **kwargs):
        defaults = {}
        defaults['C'] = kwargs.get('C', [1, 2, 5, 10, 20, 100])
        defaults['gamma'] = kwargs.get('gamma', [0.1])
        defaults['kernel'] = kwargs.get('kernel', ['linear'])
        svc = SVC()
        clf = GridSearchCV(svc, defaults)
        return clf
