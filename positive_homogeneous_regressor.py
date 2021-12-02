
import numpy as np
from sklearn.base import BaseEstimator, clone
from .utils import my_fit

# Positive Homogeneous
# https://www.jstage.jst.go.jp/article/pjsai/JSAI2020/0/JSAI2020_4Rin120/_pdf/-char/ja

class PositiveHomogeneousRegressor(BaseEstimator):
    def __init__(self, regressor=None):
        self.regressor = regressor

    def fit(self, X, y, sample_weight=None, fit_context=None):
        self.n_features_in_ = X.shape[1]
        self.regressor_ = clone(self.regressor)

        X_norm = np.sum(X.values ** 2, axis=1) ** 0.5
        X = X / (1e-37 + X_norm).reshape(-1, 1)
        y = y / (1e-37 + X_norm)
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])
        sample_weight *= X_norm ** 2

        if fit_context is not None:
            fit_context = fit_context.copy()
            X_norm = np.sum(fit_context['X_val'].values ** 2, axis=1) ** 0.5
            fit_context['X_val'] = fit_context['X_val'] / (1e-37 + X_norm).reshape(-1, 1)
            fit_context['y_val'] = fit_context['y_val'] / (1e-37 + X_norm)
            if fit_context['sample_weight_val'] is None:
                fit_context['sample_weight_val'] = np.ones(fit_context['X_val'].shape[0])
            fit_context['sample_weight_val'] *= X_norm ** 2

        my_fit(
            self.regressor_,
            X,
            y,
            sample_weight=sample_weight,
            fit_context=fit_context,
        )

        return self

    def predict(self, X):
        X_norm = np.sum(X.values ** 2, axis=1) ** 0.5
        X = X / (1e-37 + X_norm).reshape(-1, 1)

        y = self.regressor_.predict(X)
        return y * (1e-37 + X_norm)
