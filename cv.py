
import numpy as np

def _purge_idx(train_idx, val_idx, groups, purge):
    unique_groups = np.unique(groups[val_idx])
    purged_groups = unique_groups.reshape(1, -1) + np.arange(-purge, purge + 1).reshape(-1, 1)
    purged_groups = np.unique(purged_groups)
    return train_idx[~np.isin(groups[train_idx], purged_groups)]

def my_group_kfold(groups, n_splits=5, purge=12):
    if hasattr(groups, 'values'):
        groups = groups.values
    idx = np.arange(groups.size)
    g = np.sort(np.unique(groups))
    cv = []
    for i in range(n_splits):
        selected = g[i * g.size // n_splits:(i + 1) * g.size // n_splits]
        val_idx = np.isin(groups, selected)
        cv.append((
            _purge_idx(idx[~val_idx], idx[val_idx], groups, purge),
            idx[val_idx],
        ))
    return cv

def my_kfold(x, n_splits=5, purge=12):
    return my_group_kfold(np.arange(x.shape[0]), n_splits=n_splits, purge=purge)
