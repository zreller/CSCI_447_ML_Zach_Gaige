
from __future__ import annotations

import numpy as np

def minkowski_raw(A: np.ndarray, B: np.ndarray, p) -> np.ndarray:

    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"Feature-count mismatch: A has {A.shape[1]}, B has {B.shape[1]}")

    if p == 2:

        a2 = np.sum(A ** 2, axis=1)[:, None]
        b2 = np.sum(B ** 2, axis=1)[None, :]
        raw = a2 + b2 - 2.0 * (A @ B.T)
        return np.maximum(raw, 0.0) 
    diff = np.abs(A[:, None, :] - B[None, :, :])

    if p == np.inf or p == float("inf"):
        return diff.max(axis=2)
    return np.sum(diff ** p, axis=2)


def minkowski_distance(A: np.ndarray, B: np.ndarray, p) -> np.ndarray:
    raw = minkowski_raw(A, B, p)
    if p == np.inf or p == float("inf"):
        return raw
    return raw ** (1.0 / p)


def minkowski_raw_chunked(A: np.ndarray, B: np.ndarray, p, chunk: int = 500) -> np.ndarray:
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    n_q = A.shape[0]
    out = np.empty((n_q, B.shape[0]), dtype=float)
    for start in range(0, n_q, chunk):
        end = min(start + chunk, n_q)
        out[start:end] = minkowski_raw(A[start:end], B, p)
    return out


def cyclical_raw(a: np.ndarray, b: np.ndarray, period: float, p) -> np.ndarray:
 
    a = np.asarray(a, dtype=float).reshape(-1, 1)
    b = np.asarray(b, dtype=float).reshape(1, -1)
    raw_diff = np.abs(a - b)
    circ_diff = np.minimum(raw_diff, period - raw_diff)
    if p == np.inf or p == float("inf"):
        return circ_diff
    return circ_diff ** p


def cyclical_block_raw(A: np.ndarray, B: np.ndarray, periods, p) -> np.ndarray:

    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    total = np.zeros((A.shape[0], B.shape[0]))
    for j, period in enumerate(periods):
        total += cyclical_raw(A[:, j], B[:, j], period, p)
    return total

class VDM:

    def __init__(self, p: float = 1.0):
        self.p = p
        self.classes_: np.ndarray | None = None
        self._prob_tables: list[dict] = []  
        self._fallback: np.ndarray | None = None

    def fit(self, X_cat: np.ndarray, y: np.ndarray) -> "VDM":
        X_cat = np.atleast_2d(X_cat)
        y = np.asarray(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        self._fallback = np.full(n_classes, 1.0 / n_classes)
        self._prob_tables = []

        n_cols = X_cat.shape[1] if X_cat.size else 0
        for j in range(n_cols):
            col = X_cat[:, j]
            table = {}
            for v in np.unique(col):
                mask = col == v
                C_i = mask.sum()
                counts = np.bincount(y_idx[mask], minlength=n_classes)
                table[v] = counts / C_i if C_i > 0 else self._fallback.copy()
            self._prob_tables.append(table)
        return self

    def _prob_vec(self, col_idx: int, value):
        return self._prob_tables[col_idx].get(value, self._fallback)

    def delta_matrix(self, col_idx: int, query_vals: np.ndarray, ref_vals: np.ndarray) -> np.ndarray:
        q_probs = np.stack([self._prob_vec(col_idx, v) for v in query_vals])   # (n_q, C)
        r_probs = np.stack([self._prob_vec(col_idx, v) for v in ref_vals])     # (n_r, C)
        diff = np.abs(q_probs[:, None, :] - r_probs[None, :, :])              # (n_q, n_r, C)
        return np.sum(diff ** self.p, axis=2)

    def raw_distance(self, X_cat_query: np.ndarray, X_cat_ref: np.ndarray) -> np.ndarray:
        X_cat_query = np.atleast_2d(X_cat_query)
        X_cat_ref = np.atleast_2d(X_cat_ref)
        n_cols = X_cat_query.shape[1] if X_cat_query.size else 0
        total = np.zeros((X_cat_query.shape[0], X_cat_ref.shape[0]))
        for j in range(n_cols):
            total += self.delta_matrix(j, X_cat_query[:, j], X_cat_ref[:, j])
        return total

def combined_raw_distance(
    X_num_q, X_num_r,
    X_cat_q, X_cat_r,
    p,
    vdm: VDM | None = None,
    X_cyc_q=None, X_cyc_r=None, cyc_periods=None,
    chunk_size: int | None = None,
) -> np.ndarray:

    if chunk_size is not None:
        n_q_total = (X_num_q.shape[0] if X_num_q is not None and np.size(X_num_q) else
                     (X_cat_q.shape[0] if X_cat_q is not None and np.size(X_cat_q) else X_cyc_q.shape[0]))
        if n_q_total > chunk_size:
            parts = []
            for start in range(0, n_q_total, chunk_size):
                end = min(start + chunk_size, n_q_total)
                parts.append(combined_raw_distance(
                    X_num_q[start:end] if X_num_q is not None and np.size(X_num_q) else X_num_q,
                    X_num_r,
                    X_cat_q[start:end] if X_cat_q is not None and np.size(X_cat_q) else X_cat_q,
                    X_cat_r, p, vdm=vdm,
                    X_cyc_q=X_cyc_q[start:end] if X_cyc_q is not None and np.size(X_cyc_q) else X_cyc_q,
                    X_cyc_r=X_cyc_r, cyc_periods=cyc_periods, chunk_size=None,
                ))
            return np.vstack(parts)

    n_q = X_num_q.shape[0] if X_num_q is not None and np.size(X_num_q) else (
        X_cat_q.shape[0] if X_cat_q is not None and np.size(X_cat_q) else X_cyc_q.shape[0]
    )
    n_r = X_num_r.shape[0] if X_num_r is not None and np.size(X_num_r) else (
        X_cat_r.shape[0] if X_cat_r is not None and np.size(X_cat_r) else X_cyc_r.shape[0]
    )
    total = np.zeros((n_q, n_r))

    if X_num_q is not None and X_num_q.shape[1] > 0:
        total += minkowski_raw(X_num_q, X_num_r, p)

    if X_cat_q is not None and X_cat_q.shape[1] > 0:
        if vdm is None:
            raise ValueError("Categorical features present but no fitted VDM was supplied.")
        total += vdm.raw_distance(X_cat_q, X_cat_r)

    if X_cyc_q is not None and X_cyc_q.shape[1] > 0:
        total += cyclical_block_raw(X_cyc_q, X_cyc_r, cyc_periods, p)

    return total
