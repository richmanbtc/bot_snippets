import numpy as np
import cvxpy as cp
from sklearn.base import BaseEstimator, clone
from sklearn.utils import check_random_state
from sklearn.ensemble._base import _set_random_states
from .utils import my_fit

# https://proceedings.neurips.cc/paper/1996/file/f47330643ae134ca204bf6b2481fec47-Paper.pdf
ENSEMBLE_MODE_BALANCING = 'balancing'

class BaseEarlyStoppingEstimator(BaseEstimator):
    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 cv=None,
                 # max_samples=1.0,
                 # max_features=1.0,
                 ensemble_mode=None,
                 random_state=None,
                 verbose=0):

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.cv = cv
        # self.max_samples = max_samples
        # self.max_features = max_features
        self.ensemble_mode = ensemble_mode
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        # n = X.shape[0]
        random_state = check_random_state(self.random_state)
        # count = round(self.max_samples * n)
        # feature_count = round(self.max_features * X.shape[1])

        self.n_features_in_ = X.shape[1]
        self.estimators_ = []
        self.estimators_features_ = []
        if self.ensemble_mode == ENSEMBLE_MODE_BALANCING:
            self.val_errors_ = []

        cv_gen = self.cv.split(X)

        for i in range(self.n_estimators):
            train_idx, val_idx = cv_gen.__next__()

            estimator = clone(self.base_estimator)
            _set_random_states(estimator, random_state=random_state.randint(np.iinfo(np.int32).max))

            sw = None if sample_weight is None else sample_weight[train_idx]

            fit_context = {
                'X_val': indexing(X, val_idx),
                'y_val': indexing(y, val_idx),
                'sample_weight_val': None if sample_weight is None else indexing(sample_weight, val_idx),
                'early_stopping_rounds': 100,
            }

            my_fit(
                estimator,
                indexing(X, train_idx),
                indexing(y, train_idx),
                sample_weight=sw,
                fit_context=fit_context,
            )

            if self.ensemble_mode == ENSEMBLE_MODE_BALANCING:
                y_val_pred = estimator.predict(X_val)
                val_error = np.average((y_val - y_val_pred) ** 2, weights=sw_val)
                self.val_errors_.append(val_error)

            # indicies = calc_indicies(n, count, random_state)
            # feature_indicies = calc_feature_indicies(X.shape[1], feature_count, random_state)

            feature_indicies = np.arange(X.shape[1])

            self.estimators_.append(estimator)
            self.estimators_features_.append(feature_indicies)

        if self.ensemble_mode == ENSEMBLE_MODE_BALANCING:
            self.val_errors_ = np.array(self.val_errors_)

        return self

class EarlyStoppingRegressor(BaseEarlyStoppingEstimator):
    def predict(self, X):
        ys = []
        for i, estimator in enumerate(self.estimators_):
            ys.append(estimator.predict(indexing2(X, self.estimators_features_[i])))
        ys = np.array(ys)

        if self.ensemble_mode == ENSEMBLE_MODE_BALANCING:
            w = cp.Variable((len(self.estimators_), X.shape[0]))

            # 2 * w[i] * val_errors[i]
            # - w[i] * y[i] ** 2
            # + w[i] * w[j] * y[i] * y[j] -> sum(w[i] * y[i]) ** 2

            objective = cp.Minimize(
                2 * cp.sum(cp.multiply(w, np.repeat(self.val_errors_.reshape(-1, 1), X.shape[0], axis=1)))
                - cp.sum(cp.multiply(w, ys ** 2))
                + cp.sum(cp.multiply(w, ys)) ** 2
            )

            constraints = [
                0 <= w,
                cp.sum(w, axis=0) == 1,
            ]

            prob = cp.Problem(objective, constraints)
            try:
                result = prob.solve()
            except cp.error.SolverError:
                print('cvxpy solve failed. use equal weight')
                return np.mean(ys, axis=0)

            return np.sum(ys * w.value, axis=0)
        else:
            return np.mean(ys, axis=0)

class EarlyStoppingClassifier(BaseEarlyStoppingEstimator):
    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.sort(np.unique(y))
        self.n_classes_ = len(self.classes_)
        return super().fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_.take(np.argmax(proba, axis=1), axis=0)

    def predict_proba(self, X):
        class_to_idx = {}
        for i, cls in enumerate(self.classes_):
            class_to_idx[cls] = i
        proba = np.zeros(X.shape[0], self.n_classes_)

        for estimator in self.estimators_:
            if hasattr(estimator, "predict_proba"):
                p = estimator.predict_proba(X)
                for i, cls in enumerate(estimator.classes_):
                    proba[:, class_to_idx[cls]] += p[:, i]
            else:
                y_pred = estimator.predict(X)
                for i, cls in enumerate(self.classes_):
                    proba[y_pred == cls, i] += 1

        return proba / self.n_estimators

def calc_indicies(n, count, random_state):
    indicies = random_state.randint(n, size=count)
    return np.sort(indicies)

def calc_feature_indicies(n, count, random_state):
    if n == count:
        return np.arange(n)
    else:
        return random_state.choice(np.arange(n), size=count, replace=False)

def indexing(x, idx):
    if hasattr(x, 'iloc'):
        return x.iloc[idx]
    else:
        return x[idx]

def indexing2(x, idx):
    if hasattr(x, 'iloc'):
        return x.iloc[:, idx]
    else:
        return x[:, idx]
