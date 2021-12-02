
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import r2_score
from .utils import my_fit

class EraBoostXgbRegressor(BaseEstimator):
    def __init__(self, base_estimator=None, num_iterations=3, proportion=0.5, n_estimators=None):
        self.base_estimator = base_estimator
        self.num_iterations = num_iterations
        self.proportion = proportion
        self.n_estimators = n_estimators

    def fit(self, X, y, sample_weight=None, fit_context=None):
        self.n_features_in_ = X.shape[1]
        self.base_estimator_ = clone(self.base_estimator)

        my_fit(
            self.base_estimator_,
            X,
            y,
            sample_weight=sample_weight,
            fit_context=fit_context,
        )

        for iter in range(self.num_iterations - 1):
            y_pred = self.base_estimator_.predict(X)

            era_scores = []
            indicies = []
            n = y_pred.shape[0]
            m = 10
            for i in range(m):
                idx = np.arange(i * n // m, (i + 1) * n // m)
                indicies.append(idx)
                y_pred2 = indexing(y_pred, idx)
                y2 = indexing(y, idx)
                era_scores.append(r2_score(y2, y_pred2))

            score_threshold = np.quantile(era_scores, self.proportion)
            idx = []
            for i in range(m):
                if era_scores[i] <= score_threshold:
                    idx.append(indicies[i])
            idx = np.concatenate(idx)

            self.base_estimator_.n_estimators += self.n_estimators
            booster = self.base_estimator_.get_booster()
            self.base_estimator_.fit(indexing(X, idx), indexing(y, idx), xgb_model=booster)

        return self

    def predict(self, X):
        return self.base_estimator_.predict(X)

def indexing(x, idx):
    if hasattr(x, 'iloc'):
        return x.iloc[idx]
    else:
        return x[idx]
