import numpy as np
from sklearn.base import BaseEstimator, clone
from .utils import my_fit

class ClfSignRegressor(BaseEstimator):
    def __init__(self, classifier=None):
        self.classifier = classifier

    def fit(self, X, y, sample_weight=None, fit_context=None):
        self.n_features_in_ = X.shape[1]
        self.classifier_ = clone(self.classifier)

        sw = np.abs(y)
        if sample_weight is not None:
            sw *= sample_weight
        y = np.sign(y).astype('int')

        if fit_context is not None:
            fit_context = fit_context.copy()
            sw_val = np.abs(fit_context['y_val'])
            if fit_context['sample_weight_val'] is not None:
                sw_val *= fit_context['sample_weight_val']
            fit_context['y_val'] = np.sign(fit_context['y_val']).astype('int')
            fit_context['sample_weight_val'] = sw_val

        my_fit(
            self.classifier_,
            X,
            y,
            sample_weight=sample_weight,
            fit_context=fit_context,
        )

        return self

    def predict(self, X):
        proba = self.classifier_.predict_proba(X)
        return np.sum(proba * np.array(self.classifier_.classes_), axis=1)
