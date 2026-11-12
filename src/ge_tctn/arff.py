"""Minimal ARFF reader for bag-of-words text collections (dense or sparse rows)."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

_ATTR = re.compile(r"@attribute\s+('(?:[^']|\\')*'|\"[^\"]*\"|\S+)\s+(.+)$", re.IGNORECASE)


@dataclass
class Collection:
    """A document collection: term-frequency matrix X (docs x terms), labels y."""

    X: sp.csr_matrix
    y: np.ndarray
    terms: list[str]
    classes: list[str]
    name: str = ""

    @property
    def n_docs(self) -> int:
        return self.X.shape[0]


def _unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        return s[1:-1]
    return s


def load_arff(path: str) -> Collection:
    """Read an ARFF file whose last nominal attribute is the class label."""
    names: list[str] = []
    nominal: dict[int, list[str]] = {}
    rows, cols, vals, labels = [], [], [], []
    relation, in_data, n = "", False, 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("%"):
                continue
            if not in_data:
                low = line.lower()
                if low.startswith("@relation"):
                    relation = _unquote(line[9:])
                elif low.startswith("@attribute"):
                    m = _ATTR.match(line)
                    if m is None:
                        raise ValueError(f"cannot parse attribute line: {line!r}")
                    kind = m.group(2).strip()
                    if kind.startswith("{"):
                        nominal[len(names)] = [_unquote(v) for v in kind.strip("{}").split(",")]
                    names.append(_unquote(m.group(1)))
                elif low.startswith("@data"):
                    in_data = True
                    if not nominal:
                        raise ValueError("no nominal (class) attribute found")
                    cls = max(nominal)
                    class_values = nominal[cls]
                continue
            label = class_values[0]  # sparse rows omit the class when it is the first value
            if line.startswith("{"):
                for item in line.strip("{}").split(","):
                    if not item.strip():
                        continue
                    idx, val = item.strip().split(None, 1)
                    idx = int(idx)
                    if idx == cls:
                        label = _unquote(val)
                    elif float(val) != 0.0:
                        rows.append(n)
                        cols.append(idx)
                        vals.append(float(val))
            else:
                for idx, val in enumerate(line.split(",")):
                    if idx == cls:
                        label = _unquote(val)
                    elif float(val) != 0.0:
                        rows.append(n)
                        cols.append(idx)
                        vals.append(float(val))
            labels.append(label)
            n += 1
    term_idx = [i for i in range(len(names)) if i != cls]
    remap = np.full(len(names), -1)
    remap[term_idx] = np.arange(len(term_idx))
    X = sp.csr_matrix(
        (vals, (rows, remap[np.asarray(cols, dtype=int)])), shape=(n, len(term_idx))
    )
    y = np.array([class_values.index(lab) for lab in labels])
    return Collection(X, y, [names[i] for i in term_idx], class_values, relation)
