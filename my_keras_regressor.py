import numpy as np
import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import marshal
import types
import traceback

# 追加機能
# mc dropout
# early stopping

class MyKerasRegressor(KerasRegressor):
    def __init__(self, mc_count=None, split=None, fit_params={}, **kwargs):
        KerasRegressor.__init__(self, **kwargs)
        self.mc_count_ = mc_count
        self.split_ = split
        self.fit_params_ = fit_params

    def fit(self, X, y, **kwargs):
        if self.split_ is None:
            return KerasRegressor.fit(self, X, y, **self.fit_params_, **kwargs)
        else:
            train_idx, val_idx = self.split_(X)
            return KerasRegressor.fit(
                self,
                X[train_idx],
                y[train_idx],
                validation_data=(X[val_idx], y[val_idx]),
                **self.fit_params_,
                **kwargs,
            )

    def predict(self, X=None):
        if self.mc_count_ is None or self.mc_count_ == 1:
            return KerasRegressor.predict(self, X)

        ys = []

        X = tf.data.Dataset.from_tensor_slices(X)
        X = X.batch(65536)

        for i in range(self.mc_count_):
            ys.append(KerasRegressor.predict(self, X))

        return np.mean(ys, axis=0)

    def get_params(self, **params):
        res = KerasRegressor.get_params(self, **params)
        res.update({
            'mc_count': self.mc_count_,
            'split': self.split_,
            'fit_params': self.fit_params_,
        })
        return res

    def set_params(self, **params):
        self.mc_count_ = params['mc_count']
        self.split_ = params['split']
        self.fit_params_ = params['fit_params']
        params = params.copy()
        del params['mc_count']
        del params['split']
        del params['fit_params_']
        return KerasRegressor.set_params(self, **params)

    # https://stackoverflow.com/questions/8574742/how-to-pickle-an-object-of-a-class-b-having-many-variables-that-inherits-from
    def __getstate__(self):
        a_state = KerasRegressor.__getstate__(self)
        b_state = {
            'mc_count_': self.mc_count_,
            # 'split_': marshal.dumps(self.split_.__code__),
            # 'split_': self.split_,
        }
        return (a_state, b_state)

    def __setstate__(self, state):
        a_state, b_state = state
        self.mc_count_ = b_state['mc_count_']
        # code = marshal.loads(b_state['split_'])
        # self.split_ = types.FunctionType(code, globals(), "some_func_name")
        # self.split_ = b_state['split_']
        KerasRegressor.__setstate__(self, a_state)
