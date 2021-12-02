from sklearn.base import BaseEstimator, TransformerMixin

# 特徴量をコピーしてN倍にしたときに、Ridgeのalphaを変えなくて良いようにスケーリング
class RidgeFeatureCountScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = self._validate_data(X)

        return X / (X.shape[1] ** 0.5)

    def inverse_transform(self, X, y=None):
        X = self._validate_data(X)

        return X * (X.shape[1] ** 0.5)
