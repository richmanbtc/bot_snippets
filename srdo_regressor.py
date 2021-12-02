
import numpy as np
import lightgbm as lgb
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from kedro_work.utils import get_joblib_memory

memory = get_joblib_memory()

# https://arxiv.org/abs/1911.12580

class SrdoRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, estimator=None, epsilon=1e-7):
        self.estimator = estimator
        self.epsilon = epsilon

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        memory.reduce_size()
        w = calc_decorr_weight(X, self.epsilon)
        fitted = clone(self.estimator)
        fitted.fit(X, y, sample_weight=w)
        self.estimator_ = fitted

        return self

    def predict(self, X):
        return self.estimator_.predict(X)

@memory.cache
def calc_decorr_weight(X, epsilon):
    classifier = lgb.LGBMClassifier(n_jobs=-1, random_state=0)

    X_positive = []
    for i in range(X.shape[1]):
        X_positive.append(np.random.choice(X[:, i], size=X.shape[0], replace=True))
    X_positive = np.array(X_positive).transpose()

    classifier.fit(
        np.vstack([X, X_positive]),
        np.concatenate([np.zeros(X.shape[0]), np.ones(X.shape[0])])
    )
    proba = classifier.predict_proba(X)
    w = proba[:, 1] / (epsilon + proba[:, 0])

    return w / np.sum(w)
