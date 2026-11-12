"""TCTN: transductive classification through a term network (Rossi, Rezende and Lopes, 2015)."""

from __future__ import annotations

from math import fsum

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from .network import round_mantissa


def initial_term_labels(X_lab: sp.csr_matrix, y_lab: np.ndarray, n_classes: int) -> np.ndarray:
    """Eq. 2.7: share of a term's frequency in labelled documents that falls in each class."""
    onehot = np.zeros((len(y_lab), n_classes))
    onehot[np.arange(len(y_lab)), y_lab] = 1.0
    num = np.asarray(X_lab.T @ onehot)
    den = num.sum(axis=1, keepdims=True)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def llgc(W: sp.spmatrix, Y: np.ndarray, alpha: float) -> np.ndarray:
    """Fixed point of F = alpha*S*F + (1-alpha)*Y with S = D^-1/2 W D^-1/2 (Zhou et al., 2004).

    Solved exactly as (I - alpha*S) F = (1-alpha) Y with a sparse LU factorisation.
    Isolated terms keep their initial label vector.
    """
    d = np.asarray(W.sum(axis=1)).ravel()
    dinv = np.divide(1.0, np.sqrt(d), out=np.zeros_like(d), where=d > 0)
    S = sp.diags(dinv) @ W @ sp.diags(dinv)
    A = (sp.identity(W.shape[0], format="csc") - alpha * S).tocsc()
    return splu(A).solve((1.0 - alpha) * Y)


def classify_documents(X_docs: sp.csr_matrix, F: np.ndarray) -> np.ndarray:
    """Eq. 2.9: document class score = sum of term frequency x term class information.
    Scores are rounded (:func:`round_mantissa`) and ties go to the lowest class index."""
    return round_mantissa(np.asarray(X_docs @ F)).argmax(axis=1)


def _per_class(y_true, y_pred, n_classes):
    tp = np.bincount(y_true[y_true == y_pred], minlength=n_classes).astype(float)
    support = np.bincount(y_true, minlength=n_classes).astype(float)
    predicted = np.bincount(y_pred, minlength=n_classes).astype(float)
    prec = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    rec = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    return prec, rec, support / support.sum()


def weighted_f1(y_true, y_pred, n_classes: int) -> float:
    """Thesis Eqs. 4.1-4.3: harmonic mean of the support-weighted precision and recall."""
    prec, rec, w = _per_class(np.asarray(y_true), np.asarray(y_pred), n_classes)
    p, r = fsum(w * prec), fsum(w * rec)  # exactly rounded: same result on every platform
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def weighted_mean_f1(y_true, y_pred, n_classes: int) -> float:
    """Support-weighted mean of per-class F1 (scikit-learn's ``average='weighted'``)."""
    prec, rec, w = _per_class(np.asarray(y_true), np.asarray(y_pred), n_classes)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=prec + rec > 0)
    return fsum(w * f1)
