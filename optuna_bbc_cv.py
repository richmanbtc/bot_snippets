import numpy as np
import optuna
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict

class OptunaBbcCv(BaseEstimator):
    def __init__(self, create_model=None, sampler=None, n_trials=None, cv=None, scoring_y_pred=None, features=None):
        self.create_model = create_model
        self.sampler = sampler
        self.n_trials = n_trials
        self.cv = cv
        self.scoring_y_pred = scoring_y_pred
        self.features = list(features)

    def fit(self, X=None, y=None, sample_weight=None):
        cv = list(self.cv.split(X))

        y_preds = []
        def objective(trial):
            model = self.create_model(trial)

            if sample_weight is not None:
                y_pred = np.zeros(X.shape[0])
                X_filtered = self._filter_X(X)
                for train_idx, val_idx in cv:
                    model.fit(X_filtered.iloc[train_idx], y.iloc[train_idx], sample_weight=sample_weight.iloc[train_idx])
                    y_pred[val_idx] = model.predict(X_filtered.iloc[val_idx])
            else:
                y_pred = cross_val_predict(model, self._filter_X(X), y, cv=cv)

            score = self.scoring_y_pred(X, y, y_pred)
            y_preds.append(y_pred)
            return -score

        study = optuna.create_study(sampler=self._create_sampler())
        study.optimize(objective, n_trials=self.n_trials)

        model = self.create_model(study.best_trial)
        model.fit(self._filter_X(X), y, sample_weight=sample_weight)

        y_pred_oos = np.zeros(X.shape[0])
        for train_idx, val_idx in cv:
            scores = []
            for y_pred in y_preds:
                score = self.scoring_y_pred(X.iloc[train_idx], y.iloc[train_idx], y_pred[train_idx])
                scores.append(score)
            scores = np.array(scores)

            n_bests = 1
            selected_y_preds = []
            for trial_idx in np.argsort(scores)[-n_bests:]:
                selected_y_preds.append(y_preds[trial_idx][val_idx])

            y_pred_oos[val_idx] = np.mean(selected_y_preds, axis=0)

        self.study_ = study
        self.model_ = model
        self.y_preds_ = np.array(y_preds)
        self.y_pred_oos_ = y_pred_oos

        return self

    def predict(self, X=None):
        return self.model_.predict(self._filter_X(X))

    def _filter_X(self, X):
        if self.features is not None:
            return X[self.features]
        return X

    def _create_sampler(self):
        optuna_seed = 1
        if self.sampler == 'tpe':
            sampler = optuna.samplers.TPESampler(seed=optuna_seed)
        elif self.sampler == 'tpe_mv':
            sampler = optuna.samplers.TPESampler(multivariate=True, group=True, seed=optuna_seed)
        elif self.sampler == 'random':
            sampler = optuna.samplers.RandomSampler(seed=optuna_seed)
        return sampler
