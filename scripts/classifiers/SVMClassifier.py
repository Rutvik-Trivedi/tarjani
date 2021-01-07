from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_is_fitted

from .BaseClassifier import BaseClassifier


class SVMClassifier(BaseClassifier):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.requires_save = True


    def name(self):
        return 'svm_classifier'


    def modelling(self, **kwargs):
        defaults = {}
        defaults['C'] = kwargs.get('C', [1, 2, 5, 10, 20, 100])
        defaults['gamma'] = kwargs.get('gamma', [0.1])
        defaults['kernel'] = kwargs.get('kernel', ['linear'])
        svc = SVC(probability=True)
        clf = GridSearchCV(svc, defaults, verbose=1)
        return clf

    def train(self, train_X, train_y, **kwargs):
        history = self.model.fit(train_X.numpy(), train_y)
        check_is_fitted(self.model)
        return history

    def predict(self, X):
        return self.model.predict_proba(X)
