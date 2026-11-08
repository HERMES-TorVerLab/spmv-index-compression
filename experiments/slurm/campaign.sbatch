#!/bin/bash
# campaign.sbatch — every measurement the paper needs from one A100, in one job:
#   1. the device constants (device_report) and nvidia-smi,
#   2. the campaign after RCM (csr_layouts_rcm.csv, _tpr.csv, .log), which the paper reports,
#   3. the campaign in the original ordering (csr_layouts.csv ...), read only to count the matrices
#      CSR-r16 holds before reordering,
#   4. the instructions per nonzero of every kernel, read from the SASS (instructions.csv).
#
#   source experiments/slurm/env.sh          # carries the account through SBATCH_ACCOUNT
#   sbatch experiments/slurm/campaign.sbatch
#
# One GPU, not four: the measurement is a bandwidth measurement, and a job that owns a single GPU is
# the configuration the numbers are quoted for. Each ordering takes about two hours on the 74
# matrices, most of it parsing Matrix Market text and rebuilding the CSR-lp stream at every threads per
# row; the RCM permutations are computed on the first run and cached next to each .mtx.
#
# Resumable: a matrix whose "end" line is already in a log is skipped, so a job that hits the
# walltime is continued by submitting the same script again.
#
# Env:  SPMV_LIST      matrix list (default experiments/lists/all.txt)
#       REPS, WARMUP, MIN_SAMPLE_MS   timing protocol (default 20, 3, 10; keep them equal across
#                                     machines or the campaigns stop being comparable)
#
# A short test on four matrices, before spending budget on the whole set:
#   SPMV_LIST=experiments/lists/smoke.txt sbatch --qos=debug --time=00:30:00 \
#       experiments/slurm/campaign.sbatch
#
#SBATCH --job-name=spmv-layouts
# the partition comes from SBATCH_PARTITION, which env.sh exports from SPMV_PARTITION,
# so one job file serves every partition of the machine
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
# Eight hours: two orderings of about two hours each, with room for the first RCM permutations.
#SBATCH --time=08:00:00
#SBATCH --output=spmv-campaign-%j.out
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f "${ROOT}/experiments/slurm/env.sh" ]] || { echo "submit from the repository root, not from ${ROOT}"; exit 1; }
# shellcheck disable=SC1091
source "${ROOT}/experiments/slurm/env.sh"
spmv_load_modules

LIST="${SPMV_LIST:-${ROOT}/experiments/lists/all.txt}"
BENCH="${SPMV_BUILD_DIR}/bench_csr_width"
OUT="${SPMV_RESULTS_DIR}"

echo "=== job ${SLURM_JOB_ID:-none} on $(hostname), $(date '+%F %T') ==="
spmv_describe
echo "list         ${LIST}"
echo "protocol     reps ${REPS:-20}, warmup ${WARMUP:-3}, min sample ${MIN_SAMPLE_MS:-10} ms"
[[ -x "${BENCH}" ]] || { echo "no ${BENCH}: run experiments/slurm/build.sh"; exit 1; }

# The binary links cuSPARSE, and a batch job does not always get the module environment the build
# had. An older libcusparse.so.12 resolves, and the run then dies at the first matrix on a symbol
# that version does not export. ldd -r reports that before anything is measured.
missing="$(ldd -r "${BENCH}" 2>&1 | grep -E 'undefined symbol|not found' || true)"
if [[ -n "${missing}" ]]; then
  echo "${BENCH} cannot be loaded in this job:"
  echo "${missing}" | sed 's/^/  /'
  echo "  cusparse resolves to: $(ldd "${BENCH}" 2>/dev/null | awk '/cusparse/ {print $3}')"
  echo "  the build used ${SPMV_CUDA_MODULE:-<unknown>}; rebuild with experiments/slurm/build.sh"
  echo "  so that the toolkit directory is baked into the RPATH, or load the same CUDA module here"
  exit 1
fi
mkdir -p "${OUT}"

# --- the machine, measured on the GPU this job was given ------------------------------------------
{ nvidia-smi; echo; nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm,clocks.max.memory,ecc.mode.current \
                              --format=csv; } > "${OUT}/nvidia-smi.txt" 2>&1 || true
keep_memory="$(grep '^memory=' "${OUT}/device.txt" 2>/dev/null || true)"
"${SPMV_BUILD_DIR}/device_report" > "${OUT}/device.txt.new"
[[ -n "${keep_memory}" ]] && echo "${keep_memory}" >> "${OUT}/device.txt.new"
mv "${OUT}/device.txt.new" "${OUT}/device.txt"
cat "${OUT}/device.txt"
grep -q '^memory=' "${OUT}/device.txt" \
  || echo "note: add a line like 'memory=64~GB HBM2e' to ${OUT}/device.txt (the advertised capacity is not queryable)"

# --- the campaigns, RCM first since it is the one the paper reports ------------------------------
for ordering in rcm original; do
  echo "=== ordering ${ordering} === $(date '+%T')"
  OUT="${OUT}" BENCH="${BENCH}" MATRIX_ROOT="${MATRIX_ROOT}" ORDERING="${ordering}" \
  REPS="${REPS:-20}" WARMUP="${WARMUP:-3}" MIN_SAMPLE_MS="${MIN_SAMPLE_MS:-10}" \
    "${ROOT}/experiments/run_campaign.sh" "${LIST}"
done

# --- the instructions per nonzero, a property of the compiled kernel on this architecture ---------
python3 "${ROOT}/analysis/extract_instruction_counts.py" "${BENCH}" "${OUT}/instructions.csv" || true
echo "=== done $(date '+%F %T') ==="
