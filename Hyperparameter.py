
from __future__ import annotations

from typing import Callable, Sequence
import itertools
import numpy as np

from .preprocessing import k_fold_indices
from .knn import Features


def classification_error(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true != y_pred))


def mean_squared_error(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))

def tune_hyperparameters(fit_predict_factory: Callable, feats: Features, y: np.ndarray,
                          param_grid: dict, metric: Callable, lower_is_better=True,
                          inner_k=5, random_state=None):
    n = len(feats)
    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))

    all_results = []
    for combo in combos:
        params = dict(zip(keys, combo))
        fit_predict = fit_predict_factory(**params)
        fold_scores = []
        for train_idx, val_idx in k_fold_indices(n, inner_k, random_state=random_state):
            train_feats, val_feats = feats[train_idx], feats[val_idx]
            train_y, val_y = y[train_idx], y[val_idx]
            preds = fit_predict(train_feats, train_y, val_feats)
            fold_scores.append(metric(val_y, preds))
        all_results.append((params, float(np.mean(fold_scores))))

    all_results.sort(key=lambda t: t[1], reverse=not lower_is_better)
    best_params, best_score = all_results[0]
    return best_params, best_score, all_results
