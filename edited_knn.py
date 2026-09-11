
from __future__ import annotations

import numpy as np

from .knn import Features, KNNClassifier, KNNRegressor, CHUNK_SIZE
from . import distances as dist_mod


def _predict_loo_classification(feats: Features, y: np.ndarray, p, random_state=None) -> np.ndarray:
    n = len(feats)
    preds = np.empty(n, dtype=y.dtype if y.dtype != object else object)
    knn = KNNClassifier(k=2, p=p, random_state=random_state)
    knn.fit(feats, y)
    idx, _ = knn.kneighbors(Features(feats.X_num, feats.X_cat, feats.X_cyc, feats.cyc_periods), k=2)
    for i in range(n):
        cand = idx[i, 1] if idx[i, 0] == i else idx[i, 0]
        preds[i] = y[cand]
    return preds


def edited_nn_classification(feats: Features, y: np.ndarray, p=2, max_iters=50, random_state=None,
                              min_class_count=1):
    keep = np.ones(len(feats), dtype=bool)
    rng = np.random.default_rng(random_state)

    for _ in range(max_iters):
        cur_idx = np.where(keep)[0]
        cur_feats = feats[cur_idx]
        cur_y = y[cur_idx]
        if len(cur_idx) <= 2:
            break

        loo_preds = _predict_loo_classification(cur_feats, cur_y, p, random_state=rng.integers(1 << 30))
        wrong_local = loo_preds != cur_y
        removable = np.ones(len(cur_idx), dtype=bool)
        for cls in np.unique(cur_y):
            cls_mask = cur_y == cls
            n_cls = cls_mask.sum()
            n_wrong_cls = (wrong_local & cls_mask).sum()
            if n_cls - n_wrong_cls < min_class_count:
               
                wrong_cls_idx = np.where(wrong_local & cls_mask)[0]
                n_to_spare = min_class_count - (n_cls - n_wrong_cls)
                spare = rng.choice(wrong_cls_idx, size=min(n_to_spare, len(wrong_cls_idx)), replace=False)
                removable[spare] = False

        to_remove_local = wrong_local & removable
        if not to_remove_local.any():
            break
        keep[cur_idx[to_remove_local]] = False

    return feats[keep], y[keep], keep


def edited_nn_regression(feats: Features, y: np.ndarray, epsilon: float, p=2, gamma=1.0,
                          max_iters=50, random_state=None, min_remaining=5):

    keep = np.ones(len(feats), dtype=bool)
    rng = np.random.default_rng(random_state)

    for _ in range(max_iters):
        cur_idx = np.where(keep)[0]
        if len(cur_idx) <= min_remaining:
            break
        cur_feats = feats[cur_idx]
        cur_y = y[cur_idx]

        reg = KNNRegressor(k=2, p=p, gamma=gamma, random_state=random_state)
        reg.fit(cur_feats, cur_y)
        idx, _ = reg.kneighbors(cur_feats, k=2)

        loo_preds = np.empty(len(cur_idx))
        for i in range(len(cur_idx)):
            cand = idx[i, 1] if idx[i, 0] == i else idx[i, 0]
            loo_preds[i] = cur_y[cand]

        wrong_local = np.abs(loo_preds - cur_y) > epsilon
        n_would_remain = len(cur_idx) - wrong_local.sum()
        if not wrong_local.any():
            break
        if n_would_remain < min_remaining:
            budget = len(cur_idx) - min_remaining
            if budget <= 0:
                break
            worst = np.argsort(-np.abs(loo_preds - cur_y))
            chosen = worst[np.isin(worst, np.where(wrong_local)[0])][:budget]
            mask = np.zeros(len(cur_idx), dtype=bool)
            mask[chosen] = True
            wrong_local = mask

        keep[cur_idx[wrong_local]] = False

    return feats[keep], y[keep], keep


def condensed_nn_classification(feats: Features, y: np.ndarray, p=2, max_iters=50, random_state=None):
    """Hart condensing for classification. Returns (reduced_feats, reduced_y, kept_mask)."""
    n = len(feats)
    rng = np.random.default_rng(random_state)
    order = rng.permutation(n)

    z_idx = [int(order[0])]
    in_z = np.zeros(n, dtype=bool)
    in_z[z_idx[0]] = True

    for _ in range(max_iters):
        z_arr = np.array(z_idx)
        z_feats, z_y = feats[z_arr], y[z_arr]
        vdm = dist_mod.VDM(p=p if p != np.inf else 1).fit(z_feats.X_cat, z_y) if feats.X_cat.shape[1] > 0 else None

        D0 = dist_mod.combined_raw_distance(
            feats.X_num, z_feats.X_num, feats.X_cat, z_feats.X_cat, p, vdm=vdm,
            X_cyc_q=feats.X_cyc, X_cyc_r=z_feats.X_cyc, cyc_periods=feats.cyc_periods,
            chunk_size=CHUNK_SIZE,
        )  # (n, |Z|)
        nearest_j = np.argmin(D0, axis=1)
        nn_dist = D0[np.arange(n), nearest_j]
        nn_label = z_y[nearest_j]

        added_this_pass = False
        for i in order:
            i = int(i)
            if in_z[i]:
                continue
            if nn_label[i] != y[i]:
                z_idx.append(i)
                in_z[i] = True
                added_this_pass = True
                new_d = dist_mod.combined_raw_distance(
                    feats.X_num, feats.X_num[i:i + 1], feats.X_cat, feats.X_cat[i:i + 1], p, vdm=vdm,
                    X_cyc_q=feats.X_cyc, X_cyc_r=feats.X_cyc[i:i + 1], cyc_periods=feats.cyc_periods,
                    chunk_size=CHUNK_SIZE,
                )[:, 0]
                better = new_d < nn_dist
                nn_dist = np.where(better, new_d, nn_dist)
                nn_label = np.where(better, y[i], nn_label)
        if not added_this_pass:
            break

    keep = np.zeros(n, dtype=bool)
    keep[np.array(z_idx)] = True
    return feats[keep], y[keep], keep


def condensed_nn_regression(feats: Features, y: np.ndarray, epsilon: float, p=2, gamma=1.0,
                             max_iters=50, random_state=None):
    n = len(feats)
    rng = np.random.default_rng(random_state)
    order = rng.permutation(n)

    z_idx = [int(order[0])]
    in_z = np.zeros(n, dtype=bool)
    in_z[z_idx[0]] = True

    for _ in range(max_iters):
        z_arr = np.array(z_idx)
        z_feats, z_y = feats[z_arr], y[z_arr]

        D0 = dist_mod.combined_raw_distance(
            feats.X_num, z_feats.X_num, feats.X_cat, z_feats.X_cat, p, vdm=None,
            X_cyc_q=feats.X_cyc, X_cyc_r=z_feats.X_cyc, cyc_periods=feats.cyc_periods,
            chunk_size=CHUNK_SIZE,
        )
        nearest_j = np.argmin(D0, axis=1)
        nn_dist = D0[np.arange(n), nearest_j]
        nn_value = z_y[nearest_j]

        added_this_pass = False
        for i in order:
            i = int(i)
            if in_z[i]:
                continue
            if abs(nn_value[i] - y[i]) > epsilon:
                z_idx.append(i)
                in_z[i] = True
                added_this_pass = True
                new_d = dist_mod.combined_raw_distance(
                    feats.X_num, feats.X_num[i:i + 1], feats.X_cat, feats.X_cat[i:i + 1], p, vdm=None,
                    X_cyc_q=feats.X_cyc, X_cyc_r=feats.X_cyc[i:i + 1], cyc_periods=feats.cyc_periods,
                    chunk_size=CHUNK_SIZE,
                )[:, 0]
                better = new_d < nn_dist
                nn_dist = np.where(better, new_d, nn_dist)
                nn_value = np.where(better, y[i], nn_value)
        if not added_this_pass:
            break

    keep = np.zeros(n, dtype=bool)
    keep[np.array(z_idx)] = True
    return feats[keep], y[keep], keep
