"""Experimental protocol of the thesis (Section 5.1): splits, Grid-TCTN and GE-TCTN."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from math import fsum, sqrt

import numpy as np

from .arff import Collection
from .ge import GEConfig, evolve
from .grammar import Node, evaluate
from .measures import MEASURES
from .network import cooccurrence_stats, knn_graph, select_terms
from .tctn import classify_documents, initial_term_labels, llgc, weighted_f1, weighted_mean_f1

K_GRID = (1, 7, 17, 37, 57)


def mean(xs) -> float:
    """Exactly rounded mean (summation order cannot change the last bit across platforms)."""
    xs = [float(x) for x in xs]
    return fsum(xs) / len(xs)


def sd(xs) -> float:
    m = mean(xs)
    return sqrt(fsum((float(x) - m) ** 2 for x in xs) / len(xs))
ALPHA_GRID = (0.1, 0.3, 0.5, 0.7, 0.9)


@dataclass
class Split:
    labelled: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def balanced_split(y: np.ndarray, n_classes: int, fraction: float, rng) -> Split:
    """`fraction` of the collection labelled and the same amount for validation, each with
    the same number of documents per class (capped by the class size); the rest is test."""
    per_class = max(1, round(fraction * len(y) / n_classes))
    lab, val = [], []
    for c in range(n_classes):
        idx = rng.permutation(np.flatnonzero(y == c))
        n_lab = min(per_class, len(idx))
        lab.append(idx[:n_lab])
        val.append(idx[n_lab: n_lab + min(per_class, len(idx) - n_lab)])
    lab, val = np.sort(np.concatenate(lab)), np.sort(np.concatenate(val))
    test = np.setdiff1d(np.arange(len(y)), np.concatenate([lab, val]))
    return Split(lab, val, test)


class Task:
    """A collection prepared for TCTN: selected terms, pair statistics and fixed splits."""

    def __init__(self, coll: Collection, term_fraction: float = 0.1, n_splits: int = 10,
                 label_fraction: float = 0.1, graph: str = "symmetric", seed: int = 0):
        self.coll = coll
        self.n_classes = len(coll.classes)
        self.term_idx = select_terms(coll.X, term_fraction)
        self.X = coll.X[:, self.term_idx].tocsr()
        self.y = coll.y
        self.stats = cooccurrence_stats(self.X)
        self.graph = graph
        rng = np.random.default_rng(seed)
        self.splits = [balanced_split(self.y, self.n_classes, label_fraction, rng)
                       for _ in range(n_splits)]
        self.Y0 = [initial_term_labels(self.X[s.labelled], self.y[s.labelled], self.n_classes)
                   for s in self.splits]

    def run(self, M: np.ndarray, k: int, alphas, split_ids=None) -> dict:
        """Mean validation/test F1 for each alpha over the chosen splits."""
        W = knn_graph(M, k, self.graph)
        split_ids = range(len(self.splits)) if split_ids is None else split_ids
        out = {}
        for a in alphas:
            rows = []
            for s in split_ids:
                F = llgc(W, self.Y0[s], a)
                row = {}
                for part in ("validation", "test"):
                    idx = getattr(self.splits[s], part)
                    pred = classify_documents(self.X[idx], F)
                    row[part] = weighted_f1(self.y[idx], pred, self.n_classes)
                    row[part + "_mean_f1"] = weighted_mean_f1(self.y[idx], pred, self.n_classes)
                rows.append(row)
            out[a] = {key: mean([r[key] for r in rows]) for key in rows[0]}
            out[a]["test_sd"] = sd([r["test"] for r in rows])
        return out


def grid_tctn(task: Task, measures=tuple(MEASURES), ks=K_GRID, alphas=ALPHA_GRID) -> dict:
    """Grid-TCTN: choose (measure, k, alpha) by mean validation F1 over the splits."""
    table = []
    for name in measures:
        n = task.X.shape[1]
        M = np.broadcast_to(MEASURES[name](task.stats), (n, n))
        for k in ks:
            for a, res in task.run(M, k, alphas).items():
                table.append({"measure": name, "k": k, "alpha": a, **res})
    best = max(table, key=lambda r: r["validation"])
    return {"best": best, "table": table}


def ge_run(task: Task, config: GEConfig, k: int, alpha: float, run: int, seed: int,
           fitness_splits: str = "one") -> dict:
    """One GE-TCTN run for a fixed (k, alpha): evolve, then score the best function on all
    splits. Fitness is the validation F1 on split ``run`` (``fitness_splits="one"``) or the
    mean over all splits (``"all"``)."""
    s = [run % len(task.splits)] if fitness_splits == "one" else None

    def fitness(tree: Node) -> float:
        return task.run(evaluate(tree, task.stats), k, [alpha], s)[alpha]["validation"]

    res = evolve(fitness, config, seed=seed)
    final = task.run(evaluate(res.best.tree, task.stats), k, [alpha])[alpha]
    return {"k": k, "alpha": alpha, "run": run, "seed": seed, "fitness": res.best.fitness,
            "function": res.best.phenotype, "evaluations": res.evaluations,
            "history": res.history, **{"final_" + key: v for key, v in final.items()}}


def summarize_ge(runs: list[dict]) -> dict:
    """Pick the (k, alpha) with the best mean fitness over its runs (first in input order on
    ties) and report its best-by-validation run, best test score and mean test score."""
    combos: dict[tuple, list] = {}
    for r in runs:
        combos.setdefault((r["k"], r["alpha"]), []).append(r)
    table = [{"k": k, "alpha": a, "mean_fitness": mean([x["fitness"] for x in rs]),
              "runs": sorted(rs, key=lambda x: x["run"])} for (k, a), rs in combos.items()]
    chosen = max(table, key=lambda c: c["mean_fitness"])
    tests = [x["final_test"] for x in chosen["runs"]]
    return {"k": chosen["k"], "alpha": chosen["alpha"],
            "best_by_validation": max(chosen["runs"], key=lambda x: x["fitness"]),
            "best_run_test": float(max(tests)),
            "mean_test": mean(tests), "sd_test": sd(tests),
            "combos": table}


def ge_tctn(task: Task, config: GEConfig, ks=K_GRID, alphas=ALPHA_GRID, runs: int = 10,
            seed: int = 0, fitness_splits: str = "one", log=None) -> dict:
    """GE-TCTN: for each (k, alpha), run GE `runs` times (run r of pair ci uses seed
    ``seed + 1000*ci + r``) and keep the (k, alpha) with the best mean fitness."""
    results = []
    for ci, (k, a) in enumerate(itertools.product(ks, alphas)):
        for r in range(runs):
            res = ge_run(task, config, k, a, r, seed + 1000 * ci + r, fitness_splits)
            results.append(res)
            if log:
                log(f"k={k} alpha={a} run={r} fitness={res['fitness']:.4f} "
                    f"test={res['final_test']:.4f} f={res['function']}")
    return summarize_ge(results)
