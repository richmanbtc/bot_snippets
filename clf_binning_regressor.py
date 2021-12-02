import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.preprocessing import KBinsDiscretizer
from .utils import my_fit

class ClfBinningRegressor(BaseEstimator):
    def __init__(self, classifier=None, n_bins=None):
        self.classifier = classifier
        self.n_bins = n_bins

    def fit(self, X, y, sample_weight=None, fit_context=None):
        self.n_features_in_ = X.shape[1]
        self.classifier_ = clone(self.classifier)
        self.transformer_ = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='quantile')

        y = self.transformer_.fit_transform(y.reshape(-1, 1)).flatten().astype('int')

        if fit_context is not None:
            fit_context = fit_context.copy()
            fit_context['y_val'] = self.transformer_.transform(fit_context['y_val'].reshape(-1, 1)).flatten().astype('int')

        my_fit(
            self.classifier_,
            X,
            y,
            sample_weight=sample_weight,
            fit_context=fit_context,
        )

        self.class_values_ = self.transformer_.inverse_transform(np.array(self.classifier_.classes_).reshape(-1, 1)).flatten()

        return self

    def predict(self, X):
        proba = self.classifier_.predict_proba(X)
        return np.sum(proba * self.class_values_, axis=1)
