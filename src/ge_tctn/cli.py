"""Command-line interface: ``ge-tctn {info,grammar,grid,evolve}``."""

from __future__ import annotations

import argparse
import json
import sys
import time

from .arff import load_arff
from .ge import GEConfig
from .grammar import GRAMMAR_BNF
from .measures import MEASURES
from .protocol import ALPHA_GRID, K_GRID, Task, ge_tctn, grid_tctn


def _floats(s):
    return tuple(float(x) for x in s.split(","))


def _ints(s):
    return tuple(int(x) for x in s.split(","))


def _task(args) -> Task:
    coll = load_arff(args.arff)
    return Task(coll, term_fraction=args.term_fraction, n_splits=args.splits,
                graph=args.graph, seed=args.seed)


def _dump(obj, path):
    text = json.dumps(obj, indent=2)
    if path:
        with open(path, "w") as fh:
            fh.write(text + "\n")
    return text


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ge-tctn", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("grammar", help="print the grammar in BNF")
    for name in ("info", "grid", "evolve"):
        sp = sub.add_parser(name)
        sp.add_argument("arff", help="bag-of-words ARFF file, class as last nominal attribute")
        sp.add_argument("--term-fraction", type=float, default=0.1)
        sp.add_argument("--splits", type=int, default=10)
        sp.add_argument("--graph", choices=["symmetric", "mutual"], default="symmetric")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--k", type=_ints, default=K_GRID, help="comma-separated")
        sp.add_argument("--alpha", type=_floats, default=ALPHA_GRID, help="comma-separated")
        sp.add_argument("--out", help="write the JSON result here")
        if name == "evolve":
            sp.add_argument("--runs", type=int, default=10)
            sp.add_argument("--population", type=int, default=100)
            sp.add_argument("--generations", type=int, default=100)
            sp.add_argument("--fitness-splits", choices=["one", "all"], default="one",
                            help="validation sample(s) used by the fitness of each run")
    args = p.parse_args(argv)

    if args.cmd == "grammar":
        print(GRAMMAR_BNF, end="")
        return 0
    t0 = time.process_time()
    task = _task(args)
    if args.cmd == "info":
        c = task.coll
        print(f"{c.name}: {c.n_docs} documents, {len(c.terms)} terms "
              f"({task.X.shape[1]} kept), {len(c.classes)} classes")
        return 0
    if args.cmd == "grid":
        res = grid_tctn(task, tuple(MEASURES), args.k, args.alpha)
        res["cpu_seconds"] = time.process_time() - t0
        _dump(res, args.out)
        b = res["best"]
        print(f"Grid-TCTN best: {b['measure']} k={b['k']} alpha={b['alpha']} "
              f"validation={b['validation']:.4f} test={b['test']:.4f} (sd {b['test_sd']:.4f})")
        return 0
    cfg = GEConfig(population=args.population, generations=args.generations)
    res = ge_tctn(task, cfg, args.k, args.alpha, runs=args.runs, seed=args.seed,
                  fitness_splits=args.fitness_splits,
                  log=lambda m: print(m, file=sys.stderr, flush=True))
    res["config"] = {**cfg.__dict__, "fitness_splits": args.fitness_splits}
    res["cpu_seconds"] = time.process_time() - t0
    _dump(res, args.out)
    best = res["best_by_validation"]
    print(f"GE-TCTN k={res['k']} alpha={res['alpha']}: best-by-validation test="
          f"{best['final_test']:.4f}, best run test={res['best_run_test']:.4f}, "
          f"mean test={res['mean_test']:.4f} (sd {res['sd_test']:.4f})")
    print(f"function: {best['function']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
