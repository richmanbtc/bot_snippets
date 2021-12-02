import inspect
import lightgbm as lgb
import xgboost as xgb

def my_fit(model, *args, **kwargs):
    if kwargs.get('fit_context') is not None:
        fit_context = kwargs['fit_context']
        if isinstance(model, lgb.LGBMRegressor) or isinstance(model, lgb.LGBMClassifier):
            kwargs['eval_set'] = [(fit_context['X_val'], fit_context['y_val'])]
            if 'sample_weight_val' in fit_context and fit_context['sample_weight_val'] is not None:
                kwargs['eval_sample_weight'] = [fit_context['sample_weight_val']]
            kwargs['early_stopping_rounds'] = fit_context['early_stopping_rounds']
            kwargs['verbose'] = False
            del kwargs['fit_context']
            print('early stopping is used lgbm')

        if isinstance(model, xgb.XGBRegressor) or isinstance(model, xgb.XGBClassifier):
            kwargs['eval_set'] = [(fit_context['X_val'], fit_context['y_val'])]
            if 'sample_weight_val' in fit_context and fit_context['sample_weight_val'] is not None:
                kwargs['eval_sample_weight'] = [fit_context['sample_weight_val']]
            kwargs['early_stopping_rounds'] = fit_context['early_stopping_rounds']
            kwargs['verbose'] = False
            del kwargs['fit_context']
            print('early stopping is used xgb')

    argspec = inspect.getfullargspec(model.fit)
    # print(argspec)
    if 'fit_context' in kwargs and 'fit_context' not in argspec.args:
        del kwargs['fit_context']

    # print(model)
    # print(kwargs.keys())
    # print(argspec.args)
    # print(argspec)
    #
    # if 'sample_weight' in kwargs and 'sample_weight' not in argspec.args:
    #     del kwargs['sample_weight']

    return model.fit(*args, **kwargs)
