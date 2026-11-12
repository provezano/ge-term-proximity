"""The five classic proximity (interestingness) measures used by TCTN (thesis Table 4)."""

from __future__ import annotations

import numpy as np


def _safe_div(num, den):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    return np.where(np.isfinite(out), out, 0.0)


def piatetsky_shapiro(s):
    return s["pab"] - s["pa"] * s["pb"]


def mutual_information(s):
    total = 0.0
    for joint, mi, mj in (("pab", "pa", "pb"), ("paB", "pa", "pB"),
                          ("pAb", "pA", "pb"), ("pAB", "pA", "pB")):
        pxy = s[joint]
        with np.errstate(divide="ignore", invalid="ignore"):
            term = pxy * np.log2(pxy / (s[mi] * s[mj]))
        total = total + np.where(pxy > 0, np.nan_to_num(term, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    return total


def kappa(s):
    chance = s["pa"] * s["pb"] + s["pA"] * s["pB"]
    return _safe_div(s["pab"] + s["pAB"] - chance, 1.0 - chance)


def yules_q(s):
    ad, bc = s["pab"] * s["pAB"], s["pAb"] * s["paB"]
    return _safe_div(ad - bc, ad + bc)


def support(s):
    return s["pab"]


MEASURES = {
    "piatetsky_shapiro": piatetsky_shapiro,
    "mutual_information": mutual_information,
    "kappa": kappa,
    "yules_q": yules_q,
    "support": support,
}
