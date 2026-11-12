"""Print the validation tables (thesis Tables 6 and 10 vs this reimplementation) from JSON results.

usage: python summarize.py RESULTS_DIR
RESULTS_DIR holds grid_<collection>.json (ge-tctn grid --out) and, for the full-budget GE runs,
ge_<collection>_full.json (run_units.py collect).
"""

import json
import sys
from pathlib import Path

# Thesis: Grid-TCTN mean (Table 6) and sd (Table 10); GE-TCTN best of 10 runs, mean and sd (Table 6).
THESIS = {
    "CSTR": (0.7443, 0.0336, 0.7760, 0.7467, 0.0110),
    "tr23": (0.4327, 0.1108, 0.5305, 0.5112, 0.0246),
    "tr12": (0.4023, 0.2373, 0.4993, 0.4391, 0.0300),
    "tr11": (0.2306, 0.1164, 0.3983, 0.2664, 0.0604),
    "oh15": (0.7194, 0.0149, 0.7233, 0.7170, 0.0062),
}


def main(folder: str) -> None:
    folder = Path(folder)
    grid = {}
    print("| collection | thesis Grid-TCTN (sd) | ours (sd) | setting chosen here |")
    print("|---|---|---|---|")
    for name, (tg, tsd, *_) in THESIS.items():
        b = json.loads((folder / f"grid_{name}.json").read_text())["best"]
        grid[name] = b["test"]
        print(f"| {name} | {tg:.4f} ({tsd:.4f}) | {b['test']:.4f} ({b['test_sd']:.4f}) | "
              f"{b['measure']}, k={b['k']}, alpha={b['alpha']} |")
    print()
    print("| collection | thesis GE best / mean (sd) | thesis gain best / mean | ours GE best by "
          "validation / best by test / mean (sd) | our gain best-by-val / best-by-test / mean "
          "| chosen k, alpha | units | best function (by validation) |")
    print("|---|---|---|---|---|---|---|---|")
    for name, (tg, _, tb, tm, ts) in THESIS.items():
        path = folder / f"ge_{name}_full.json"
        if not path.exists():
            continue
        ge = json.loads(path.read_text())
        bv, bt, m = ge["best_by_validation"]["final_test"], ge["best_run_test"], ge["mean_test"]
        g = grid[name]
        print(f"| {name} | {tb:.4f} / {tm:.4f} ({ts:.4f}) | {tb - tg:+.4f} / {tm - tg:+.4f} | "
              f"{bv:.4f} / {bt:.4f} / {m:.4f} ({ge['sd_test']:.4f}) | "
              f"{bv - g:+.4f} / {bt - g:+.4f} / {m - g:+.4f} | {ge['k']}, {ge['alpha']} | "
              f"{ge['units']} | `{ge['best_by_validation']['function']}` |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
