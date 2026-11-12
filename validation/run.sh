#!/usr/bin/env bash
# Reproduce the validation table of README.md. Expects the ARFF files from fetch_data.sh
# in $1 (default: data) and writes JSON results to $2 (default: runs).
# About 18 minutes of CPU in total (single process).
set -euo pipefail
data="${1:-data}"; out="${2:-runs}"
mkdir -p "$out"
for f in CSTR.arff tr23.wc.arff tr12.wc.arff tr11.wc.arff oh15.wc.arff; do
  ge-tctn grid "$data/$f" --out "$out/grid_${f%%.*}.json"
done
# GE-TCTN with a reduced budget (the thesis used population 100, 100 generations,
# the full 5 x 5 grid of k and alpha, and 10 runs per grid point).
ge-tctn evolve "$data/CSTR.arff" --alpha 0.1,0.5,0.9 --population 50 --generations 30 \
  --out "$out/ge_CSTR.json"
ge-tctn evolve "$data/oh15.wc.arff" --k 1,7,17 --alpha 0.1,0.5,0.9 --population 50 --generations 30 \
  --out "$out/ge_oh15.json"
ge-tctn evolve "$data/tr23.wc.arff" --k 1 --alpha 0.1,0.5,0.9 --population 30 --generations 20 \
  --out "$out/ge_tr23.json"
# Fitness averaged over the ten samples, reduced and thesis-sized GE.
ge-tctn evolve "$data/CSTR.arff" --k 1 --alpha 0.1,0.5,0.9 --population 50 --generations 30 \
  --fitness-splits all --out "$out/ge_CSTR_all.json"
ge-tctn evolve "$data/CSTR.arff" --k 1 --alpha 0.7 --fitness-splits all --out "$out/ge_CSTR_full.json"
python summarize.py "$out"
