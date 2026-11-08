#!/usr/bin/env bash
# smoke.sh - build the benchmark, fetch two small matrices, measure them and check the results.
# About ten minutes on one GPU. Writes under results/smoke, which it replaces.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
MATRIX_ROOT="${MATRIX_ROOT:-${ROOT}/matrices}"
cmake -B build -S . >/dev/null
cmake --build build --target bench_csr_width device_report -j "${JOBS:-8}" 2>&1 | grep -E "error|Built target" || true
# The two matrices travel with the public repository, so this test needs no download.
for matrix_path in structural/dawson5/dawson5 cfd/rim/rim; do
  bundled_matrix="data/smoke_matrices/$(basename "${matrix_path}").mtx.gz"
  if [[ ! -f "${MATRIX_ROOT}/${matrix_path}.mtx" && -f "${bundled_matrix}" ]]; then
    mkdir -p "$(dirname "${MATRIX_ROOT}/${matrix_path}")"
    gunzip -c "${bundled_matrix}" > "${MATRIX_ROOT}/${matrix_path}.mtx"
  fi
done
python3 matrices/download.py --root "${MATRIX_ROOT}" --names dawson5,rim
(cd "${MATRIX_ROOT}" && grep -E '/(dawson5|rim)\.mtx$' "${ROOT}/data/matrices.sha256" | sha256sum -c -)
printf '%s\n' matrices/structural/dawson5/dawson5.mtx matrices/cfd/rim/rim.mtx > build/smoke.txt
rm -rf results/smoke
OUT=results/smoke BENCH=build/bench_csr_width MATRIX_ROOT="${MATRIX_ROOT}" ORDERING=rcm REPS=5 WARMUP=1 MIN_SAMPLE_MS=2 \
  experiments/run_campaign.sh build/smoke.txt
python3 analysis/compare_campaign.py results/smoke/csr_layouts_rcm.csv
