
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

import distances as dist_mod

CHUNK_SIZE = 300


@dataclass
class Features:
    X_num: np.ndarray
    X_cat: np.ndarray
    X_cyc: np.ndarray
    cyc_periods: list = field(default_factory=list)

    def __len__(self):
        return self.X_num.shape[0]

    def __getitem__(self, idx) -> "Features":
        return Features(self.X_num[idx], self.X_cat[idx], self.X_cyc[idx], self.cyc_periods)

    @staticmethod
    def empty_like(n_rows: int) -> np.ndarray:
        return np.zeros((n_rows, 0))


class KNNClassifier:
    def __init__(self, k: int = 5, p=2, random_state=None):
        self.k = k
        self.p = p
        self.random_state = random_state

    def fit(self, feats: Features, y: np.ndarray) -> "KNNClassifier":
        self.feats_ = feats
        self.y_ = np.asarray(y)
        self.vdm_ = None
        if feats.X_cat.shape[1] > 0:
            self.vdm_ = dist_mod.VDM(p=self.p if self.p != np.inf else 1).fit(feats.X_cat, self.y_)
        self._rng = np.random.default_rng(self.random_state)
        return self

    def _raw_distances(self, q: Features) -> np.ndarray:
        return dist_mod.combined_raw_distance(
            q.X_num, self.feats_.X_num,
            q.X_cat, self.feats_.X_cat,
            self.p, vdm=self.vdm_,
            X_cyc_q=q.X_cyc, X_cyc_r=self.feats_.X_cyc, cyc_periods=q.cyc_periods,
            chunk_size=CHUNK_SIZE,
        )

    def kneighbors(self, q: Features, k: int | None = None):
        k = k or self.k
        n_train = len(self.feats_)
        k = min(k, n_train)
        D = self._raw_distances(q)
        idx = np.argsort(D, axis=1, kind="stable")[:, :k]
        rows = np.arange(D.shape[0])[:, None]
        return idx, D[rows, idx]

    def predict(self, q: Features) -> np.ndarray:
        idx, _ = self.kneighbors(q)
        neighbor_labels = self.y_[idx]  # (n_q, k)
        preds = []
        for row in neighbor_labels:
            values, counts = np.unique(row, return_counts=True)
            top = values[counts == counts.max()]
            preds.append(top[0] if len(top) == 1 else top[self._rng.integers(len(top))])
        dtype = object if (len(preds) and isinstance(preds[0], str)) else (self.y_.dtype if len(preds) else float)
        return np.array(preds, dtype=dtype)


class KNNRegressor:
    def __init__(self, k: int = 5, p=2, gamma: float = 1.0, random_state=None):
        self.k = k
        self.p = p
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, feats: Features, y: np.ndarray) -> "KNNRegressor":
        self.feats_ = feats
        self.y_ = np.asarray(y, dtype=float)
        self._rng = np.random.default_rng(self.random_state)
        return self

    def _raw_distances(self, q: Features, p) -> np.ndarray:
        return dist_mod.combined_raw_distance(
            q.X_num, self.feats_.X_num,
            q.X_cat, self.feats_.X_cat,
            p, vdm=None,
            X_cyc_q=q.X_cyc, X_cyc_r=self.feats_.X_cyc, cyc_periods=q.cyc_periods,
            chunk_size=CHUNK_SIZE,
        )

    def kneighbors(self, q: Features, k: int | None = None):
        k = k or self.k
        n_train = len(self.feats_)
        k = min(k, n_train)
        D = self._raw_distances(q, self.p)
        idx = np.argsort(D, axis=1, kind="stable")[:, :k]
        rows = np.arange(D.shape[0])[:, None]
        return idx, D[rows, idx]

    def predict(self, q: Features) -> np.ndarray:
        idx, _ = self.kneighbors(q)
        preds = np.empty(idx.shape[0])
        for i, neigh_idx in enumerate(idx):
            neigh_feats = self.feats_[neigh_idx]
            q_row = q[i:i + 1]
            l2_raw = dist_mod.combined_raw_distance(
                q_row.X_num, neigh_feats.X_num,
                q_row.X_cat, neigh_feats.X_cat,
                p=2, vdm=None,
                X_cyc_q=q_row.X_cyc, X_cyc_r=neigh_feats.X_cyc, cyc_periods=q_row.cyc_periods,
            )
            l2 = np.sqrt(l2_raw[0])
            weights = np.exp(-self.gamma * l2)
            y_neigh = self.y_[neigh_idx]
            if weights.sum() <= 1e-12:
                preds[i] = y_neigh.mean()
            else:
                preds[i] = np.sum(weights * y_neigh) / np.sum(weights)
        return preds
