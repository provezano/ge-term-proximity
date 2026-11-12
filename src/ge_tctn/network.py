"""Term network: term selection, co-occurrence statistics and k-NN graphs."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def select_terms(X: sp.csr_matrix, fraction: float = 0.1) -> np.ndarray:
    """Indices of the top `fraction` of terms ranked by summed TF-IDF over all documents."""
    n_docs, n_terms = X.shape
    df = np.asarray((X > 0).sum(axis=0)).ravel()
    idf = np.log(n_docs / np.maximum(df, 1))
    score = np.asarray(X.sum(axis=0)).ravel() * idf
    n_keep = max(2, int(np.ceil(fraction * n_terms)))
    order = np.lexsort((np.arange(n_terms), -score))  # stable: ties broken by column index
    return np.sort(order[:n_keep])


def cooccurrence_stats(X: sp.csr_matrix) -> dict[str, np.ndarray]:
    """Document-level contingency probabilities for every pair of terms.

    Keys follow the grammar terminals: ``a`` is t_i, ``b`` is t_j, upper case is negation.
    ``pa``/``pA`` have shape (n, 1) and ``pb``/``pB`` shape (1, n) so they broadcast;
    the joint cells have shape (n, n).
    """
    B = (X > 0).astype(np.float64).tocsc()
    n_docs = B.shape[0]
    p = np.asarray(B.sum(axis=0)).ravel() / n_docs
    pab = (B.T @ B).toarray() / n_docs
    pa, pb = p[:, None], p[None, :]
    paB = pa - pab
    pAb = pb - pab
    pAB = 1.0 - pa - pb + pab
    return {"pab": pab, "paB": paB, "pAb": pAb, "pAB": pAB,
            "pa": pa, "pA": 1.0 - pa, "pb": pb, "pB": 1.0 - pb}


def round_mantissa(x: np.ndarray, bits: int = 30) -> np.ndarray:
    """Round to ``bits`` mantissa bits so that last-bit differences between platforms' math
    libraries do not change comparisons. Infinities and NaN pass through."""
    m, e = np.frexp(x)
    return np.ldexp(np.round(m * 2.0**bits) / 2.0**bits, e)


def knn_graph(M: np.ndarray, k: int, kind: str = "symmetric") -> sp.csr_matrix:
    """Weighted k-NN term graph from a proximity matrix (larger = closer).

    Non-finite proximities never become neighbours; negative weights are clipped to zero.
    Values are compared after :func:`round_mantissa`, and ties at the k-th place go to the
    lowest term index, so the graph does not depend on the platform.
    ``symmetric``: edge if j is in kNN(i) or i in kNN(j); ``mutual``: both.
    """
    n = M.shape[0]
    M = np.array(M, dtype=np.float64, copy=True)
    M[~np.isfinite(M)] = -np.inf
    np.fill_diagonal(M, -np.inf)
    M = round_mantissa(M)
    k = min(k, n - 1)
    neg = -M
    kth = np.partition(neg, k - 1, axis=1)[:, k - 1: k]
    tie = neg == kth
    need = k - (neg < kth).sum(axis=1, keepdims=True)
    rows, cols = np.nonzero((neg < kth) | (tie & (np.cumsum(tie, axis=1) <= need)))
    w = M[rows, cols]
    w = np.where(np.isfinite(w), np.maximum(w, 0.0), 0.0)
    A = sp.csr_matrix((w, (rows, cols)), shape=(n, n))
    A.eliminate_zeros()
    if kind == "symmetric":
        W = A.maximum(A.T)
    elif kind == "mutual":
        W = A.minimum(A.T)
    else:
        raise ValueError(f"unknown graph kind {kind!r}")
    return W.tocsr()
