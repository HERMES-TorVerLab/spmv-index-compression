#!/usr/bin/env bash
# run_campaign.sh — the campaign of the paper: CSR-i32, CSR-i16/i8, CSR-r16/r8, CSR-lp and cuSPARSE at
# FP64, FP32 and FP16, every layout at every threads per row, on every matrix of a list.
#
# Every timing sample runs enough consecutive products to last MIN_SAMPLE_MS, so sub-millisecond
# kernels are not biased by launch and synchronisation costs; the reported time is the median over
# REPS samples divided by the number of products per sample.
#
# Usage:   experiments/run_campaign.sh [matrix list]     (default experiments/lists/all.txt)
# Env:     OUT          output directory           (default results/local)
#          BENCH        benchmark binary           (default build/bench_csr_width)
#          MATRIX_ROOT  directory holding <category>/<name>/<name>.mtx (default matrices/)
#          REPS, WARMUP, MIN_SAMPLE_MS             (default 20, 3, 10)
#          ORDERING     rcm | original             (default rcm)
#          PRECISION    all | fp64 | fp32 | fp16   (default all)
# Output:  $OUT/csr_layouts_rcm.csv, .log and _tpr.csv (every threads per row of every layout), or
#          csr_layouts.* with ORDERING=original
#
# The paper reports the RCM ordering, P A P^T, and reads the original one only to count the matrices
# CSR-r16 holds before reordering. The permutation is computed once and cached next to the .mtx.
# Resumable: a matrix whose "end" line is in the log is skipped.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIST="${1:-${ROOT}/experiments/lists/all.txt}"
OUT="${OUT:-${ROOT}/results/local}"
BENCH="${BENCH:-${ROOT}/build/bench_csr_width}"
MATRIX_ROOT="${MATRIX_ROOT:-${ROOT}/matrices}"
REPS="${REPS:-20}"
WARMUP="${WARMUP:-3}"
MIN_SAMPLE_MS="${MIN_SAMPLE_MS:-10}"
ORDERING="${ORDERING:-rcm}"
PRECISION="${PRECISION:-all}"
# A reordered run writes its own pair of files: the two orderings are different measurements of the
# same matrix and mixing them in one CSV would make the resume logic skip the second.
suffix=""
[[ "${ORDERING}" != "original" ]] && suffix="_${ORDERING}"
CSV="${OUT}/csr_layouts${suffix}.csv"
LOG="${OUT}/csr_layouts${suffix}.log"
# the time of every layout at every threads per row, so that layouts can be compared at the same one
TPR_CSV="${OUT}/csr_layouts${suffix}_tpr.csv"

[[ -x "${BENCH}" ]] || { echo "build the benchmark first (cmake --build build), or set BENCH"; exit 1; }
mkdir -p "${OUT}"

# The resume below skips a matrix whose "end" line is already in the log, which is what makes a
# killed campaign cheap to restart. It cannot tell "already measured" from "already measured with a
# different binary", and a clone or a pull restores a log that is complete, so a rebuilt benchmark
# would silently measure nothing and the old numbers would pass for new ones. If the binary is newer
# than the log, say so and stop.
if [[ -f "${LOG}" && "${BENCH}" -nt "${LOG}" ]]; then
  echo "${BENCH} is newer than ${LOG}."
  echo "The log already marks matrices as done, so they would all be skipped and the campaign would"
  echo "report the measurements of the previous binary. Move the old results aside and start clean:"
  echo "  mkdir -p ${OUT}/previous && mv ${LOG} ${CSV} ${OUT}/previous/"
  echo "or set SPMV_FORCE_RESUME=1 if you really are resuming a run of this same binary."
  [[ "${SPMV_FORCE_RESUME:-0}" == "1" ]] || exit 1
fi
touch "${LOG}"

while read -r matrix_entry; do
  [[ -z "${matrix_entry}" ]] && continue
  matrix_file="${MATRIX_ROOT}/${matrix_entry#matrices/}"
  name="$(basename "${matrix_file}" .mtx)"
  [[ -f "${matrix_file}" ]] || { echo "[missing] ${matrix_file}" | tee -a "${LOG}"; continue; }
  if grep -q "] end ${name}$" "${LOG}"; then
    echo "[skip] ${name}" >> "${LOG}"
    continue
  fi
  echo "[$(date '+%F %T')] start ${name}" >> "${LOG}"
  "${BENCH}" --matrix "${matrix_file}" --ordering "${ORDERING}" \
    --reps "${REPS}" --warmup "${WARMUP}" \
    --min-sample-ms "${MIN_SAMPLE_MS}" --precision "${PRECISION}" --csv "${CSV}" --tpr-csv "${TPR_CSV}" 2>&1 | grep -a -v "^ \*\* On entry" >> "${LOG}"
  echo "[$(date '+%F %T')] end ${name}" >> "${LOG}"
done < "${LIST}"
echo "[$(date '+%F %T')] campaign complete" >> "${LOG}"
