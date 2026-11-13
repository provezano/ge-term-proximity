#!/usr/bin/env bash
# Reproduce the validation tables of README.md. Expects the ARFF files from fetch_data.sh
# in $1 (default: data) and writes results to $2 (default: results).
# Grid-TCTN takes about a minute of CPU per collection. GE-TCTN at the thesis budget
# (250 GE runs per collection) took 15 (CSTR), 36 (Oh15) and 80 (Tr23) CPU-hours here, on a mix of
# laptop cores and cloud vCPUs; set JOBS to the number of cores to use. Finished runs are skipped on restart.
set -euo pipefail
data="${1:-data}"; out="${2:-results}"; jobs="${JOBS:-4}"
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$out"
for f in CSTR.arff tr23.wc.arff tr12.wc.arff tr11.wc.arff oh15.wc.arff; do
  ge-tctn grid "$data/$f" --out "$out/grid_${f%%.*}.json"
done
python "$here/run_units.py" plan "$out/units.jsonl" CSTR.arff oh15.wc.arff tr23.wc.arff
python "$here/run_units.py" run "$data" "$out/units.jsonl" "$out/units" --jobs "$jobs"
python "$here/run_units.py" collect "$out/units" "$out"
python "$here/summarize.py" "$out"
