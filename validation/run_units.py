"""Run GE-TCTN at full budget as independent, resumable units (one GE run per unit).

  python run_units.py plan UNITS.jsonl CSTR.arff oh15.wc.arff ...
      write one line per (collection, k, alpha, run) with the seed ge_tctn() would use
  python run_units.py run DATA_DIR UNITS.jsonl OUT_DIR [--jobs 4] [--select i/n]
      run every unit whose OUT_DIR/<unit>.json does not exist yet (atomic writes)
  python run_units.py collect OUT_DIR RESULTS_DIR
      write RESULTS_DIR/ge_<collection>_full.json (same format as `ge-tctn evolve --out`)

Settings are the thesis's: population 100, 100 generations, the 5 x 5 grid of k and alpha,
10 runs per pair, fitness averaged over the ten validation samples.
"""

import argparse
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy

import ge_tctn
from ge_tctn.arff import load_arff
from ge_tctn.ge import GEConfig
from ge_tctn.protocol import ALPHA_GRID, K_GRID, Task, ge_run, summarize_ge

CONFIG = GEConfig(population=100, generations=100)
FITNESS_SPLITS = "all"
RUNS = 10


def unit_id(u: dict) -> str:
    return f"{u['collection']}_k{u['k']}_a{u['alpha']}_r{u['run']}"


def source_hash() -> str:
    h = hashlib.sha256()
    for f in sorted(Path(ge_tctn.__file__).parent.glob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def plan(args) -> None:
    with open(args.units, "w") as fh:
        for f in args.files:
            name = f.split(".")[0]
            for ci, (k, a) in enumerate(itertools.product(K_GRID, ALPHA_GRID)):
                for r in range(RUNS):
                    fh.write(json.dumps({"collection": name, "file": f, "k": k, "alpha": a,
                                         "run": r, "seed": 1000 * ci + r}) + "\n")


_TASKS: dict = {}


def _run_one(job) -> str:
    data_dir, out_dir, u = job
    if u["file"] not in _TASKS:
        _TASKS[u["file"]] = Task(load_arff(os.path.join(data_dir, u["file"])), seed=0)
    t0 = time.process_time()
    res = ge_run(_TASKS[u["file"]], CONFIG, u["k"], u["alpha"], u["run"], u["seed"],
                 FITNESS_SPLITS)
    res.update(collection=u["collection"], cpu_seconds=time.process_time() - t0,
               host=platform.node(), machine=platform.machine(), python=platform.python_version(),
               numpy=np.__version__, scipy=scipy.__version__, source=source_hash(),
               config={**CONFIG.__dict__, "fitness_splits": FITNESS_SPLITS})
    path = os.path.join(out_dir, unit_id(u) + ".json")
    with open(path + ".tmp", "w") as fh:
        json.dump(res, fh)
    os.replace(path + ".tmp", path)
    return unit_id(u)


def run(args) -> None:
    os.makedirs(args.out, exist_ok=True)
    units = [json.loads(line) for line in open(args.units) if line.strip()]
    if args.select:
        i, n = map(int, args.select.split("/"))
        units = units[i::n]
    todo = [u for u in units if not os.path.exists(os.path.join(args.out, unit_id(u) + ".json"))]
    print(f"{len(units)} units, {len(todo)} to run, {args.jobs} jobs", flush=True)
    jobs = [(args.data, args.out, u) for u in todo]
    if args.jobs == 1:
        for j in jobs:
            print(_run_one(j), flush=True)
        return
    with mp.get_context("spawn").Pool(args.jobs, maxtasksperchild=50) as pool:
        for uid in pool.imap_unordered(_run_one, jobs):
            print(uid, flush=True)


def collect(args) -> None:
    by_coll: dict = {}
    for f in sorted(Path(args.out).glob("*.json")):
        r = json.loads(f.read_text())
        by_coll.setdefault(r["collection"], []).append(r)
    os.makedirs(args.results, exist_ok=True)
    order = {pair: i for i, pair in enumerate(itertools.product(K_GRID, ALPHA_GRID))}
    for name, runs in by_coll.items():
        expected = len(K_GRID) * len(ALPHA_GRID) * RUNS
        runs.sort(key=lambda r: (order[(r["k"], r["alpha"])], r["run"]))
        res = summarize_ge(runs)
        res["config"] = runs[0]["config"]
        res["units"] = len(runs)
        res["complete"] = len(runs) == expected
        res["cpu_seconds"] = float(sum(r["cpu_seconds"] for r in runs))
        res["hosts"] = sorted({(r["machine"], r["numpy"], r["scipy"], r["source"]) for r in runs})
        out = Path(args.results) / f"ge_{name}_full.json"
        out.write_text(json.dumps(res, indent=1) + "\n")
        print(f"{name}: {len(runs)}/{expected} units -> {out}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("plan")
    a.add_argument("units")
    a.add_argument("files", nargs="+")
    a = sub.add_parser("run")
    a.add_argument("data")
    a.add_argument("units")
    a.add_argument("out")
    a.add_argument("--jobs", type=int, default=1)
    a.add_argument("--select", help="i/n: run every n-th unit starting at i")
    a = sub.add_parser("collect")
    a.add_argument("out")
    a.add_argument("results")
    args = p.parse_args(argv)
    {"plan": plan, "run": run, "collect": collect}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
