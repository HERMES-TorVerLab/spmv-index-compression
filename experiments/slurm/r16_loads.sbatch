#!/bin/bash
# r16_loads.sbatch - does CSR-r16 lose rate because each load carries fewer bytes?
#
# Three control builds change how the offsets reach the kernel and nothing else, and each is timed with
# the campaign build on the same matrix, so that the four see the same machine:
#   bench_csr_width            the campaign build
#   bench_csr_width_r16lane    offsets laid out lane by lane, one 16-bit load per nonzero
#   bench_csr_width_r16pair    the same stream, one 32-bit load per two nonzeros of the lane
#   bench_csr_width_r16wide    offsets stored in 32 bits, the bytes per load of CSR-i32
#
#   source experiments/slurm/env.sh
#   experiments/slurm/build.sh
#   sbatch experiments/slurm/r16_loads.sbatch
#   python3 analysis/r16_loads.py ${SPMV_RESULTS_DIR}/r16_loads
#
# Env:  SPMV_LIST  matrix list (default experiments/lists/r16_loads.txt)
# Output: ${SPMV_RESULTS_DIR}/r16_loads/<binary>/csr_layouts_rcm.csv, .log and _tpr.csv, RCM ordering.
#
#SBATCH --job-name=spmv-r16loads
# the partition comes from SBATCH_PARTITION, which env.sh exports from SPMV_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=180G
#SBATCH --time=16:00:00
#SBATCH --output=spmv-r16loads-%j.out

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f "${ROOT}/experiments/slurm/env.sh" ]] || { echo "submit from the repository root, not from ${ROOT}"; exit 1; }
# shellcheck disable=SC1091
source "${ROOT}/experiments/slurm/env.sh"
spmv_load_modules

LIST="${SPMV_LIST:-${ROOT}/experiments/lists/r16_loads.txt}"
BINARIES=(bench_csr_width bench_csr_width_r16lane bench_csr_width_r16pair bench_csr_width_r16wide)
OUT_ROOT="${SPMV_RESULTS_DIR}/r16_loads"

echo "=== job ${SLURM_JOB_ID:-none} on $(hostname), $(date '+%F %T') ==="
spmv_describe
echo "list         ${LIST}"
for binary in "${BINARIES[@]}"; do
  bench="${SPMV_BUILD_DIR}/${binary}"
  [[ -x "${bench}" ]] || { echo "no ${bench}: run experiments/slurm/build.sh"; exit 1; }
  missing="$(ldd -r "${bench}" 2>&1 | grep -E 'undefined symbol|not found' || true)"
  [[ -z "${missing}" ]] || { echo "${bench} cannot be loaded in this job:"; echo "${missing}"; exit 1; }
  mkdir -p "${OUT_ROOT}/${binary}"
done
"${SPMV_BUILD_DIR}/device_report" > "${OUT_ROOT}/device.txt"

# One matrix at a time through the four binaries, so a drift of the machine affects them alike.
# run_campaign.sh skips a matrix its log already marks as done, so the job resumes.
while IFS= read -r entry; do
  [[ -z "${entry}" || "${entry}" == \#* ]] && continue
  printf '%s\n' "${entry}" > "${OUT_ROOT}/current.txt"
  for binary in "${BINARIES[@]}"; do
    OUT="${OUT_ROOT}/${binary}" BENCH="${SPMV_BUILD_DIR}/${binary}" MATRIX_ROOT="${MATRIX_ROOT}" ORDERING="rcm" \
    PRECISION="all" REPS="${REPS:-20}" WARMUP="${WARMUP:-3}" MIN_SAMPLE_MS="${MIN_SAMPLE_MS:-10}" \
    SPMV_FORCE_RESUME=1 "${ROOT}/experiments/run_campaign.sh" "${OUT_ROOT}/current.txt"
  done
done < "${LIST}"
rm -f "${OUT_ROOT}/current.txt"
echo "=== done $(date '+%F %T') ==="
