"""Print the validation tables (thesis Table 6 vs this reimplementation) from JSON results."""

import json
import sys
from pathlib import Path

# Thesis Table 6 and Table 10 (weighted F1): Grid-TCTN mean and sd, GE-TCTN best, GE-TCTN mean, sd.
THESIS = {
    "CSTR": (0.7443, 0.0336, 0.7760, 0.7467, 0.0110),
    "tr23": (0.4327, 0.1108, 0.5305, 0.5112, 0.0246),
    "tr12": (0.4023, 0.2373, 0.4993, 0.4391, 0.0300),
    "tr11": (0.2306, 0.1164, 0.3983, 0.2664, 0.0604),
    "oh15": (0.7194, 0.0149, 0.7233, 0.7170, 0.0062),
}


def main(folder: str) -> None:
    folder = Path(folder)
    print("| collection | thesis Grid-TCTN (sd) | ours (sd) | ours, chosen setting |")
    print("|---|---|---|---|")
    for name, (tg, tsd, *_) in THESIS.items():
        b = json.loads((folder / f"grid_{name}.json").read_text())["best"]
        print(f"| {name} | {tg:.4f} ({tsd:.4f}) | {b['test']:.4f} ({b['test_sd']:.4f}) | "
              f"{b['measure']}, k={b['k']}, alpha={b['alpha']} |")
    print()
    print("| run | GE settings | thesis best / mean (sd) | ours best by validation | "
          "ours max over runs | ours mean (sd) |")
    print("|---|---|---|---|---|---|")
    for path in sorted(folder.glob("ge_*.json")):
        name = path.stem[3:].split("_")[0]
        ge = json.loads(path.read_text())
        c = ge["config"]
        _, _, tb, tm, ts = THESIS[name]
        setting = (f"pop {c['population']}, gen {c['generations']}, "
                   f"{len(ge['combos'])} (k, alpha), fitness on {c.get('fitness_splits', 'one')} "
                   f"sample(s); chose k={ge['k']}, alpha={ge['alpha']}")
        print(f"| {path.stem[3:]} | {setting} | {tb:.4f} / {tm:.4f} ({ts:.4f}) | "
              f"{ge['best_by_validation']['final_test']:.4f} | {ge['best_run_test']:.4f} | "
              f"{ge['mean_test']:.4f} ({ge['sd_test']:.4f}) |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs")
