#!/bin/bash
# fp16_controls.sbatch - control experiments on what CSR-r16 loses against CSR-i32 at FP16.
#
# The campaign stores FP16 values and vectors and accumulates in FP32. At that precision CSR-r16
# keeps only part of the byte ratio, and two causes other than the bytes are tested here, each with
# a build that changes one thing and leaves every other kernel bit-identical:
#   bench_csr_width            the campaign build, as the reference of this same run
#   bench_csr_width_fp16acc    accumulates in FP16, one HFMA2 per nonzero instead of two widenings
#                              and an FFMA, which tells what the widening costs
#   bench_csr_width_r16unroll  unrolls the CSR-r16 loop at FP16 by 16, as nvcc does for CSR-i32,
#                              instead of the 4 it chooses there, which tells what the loads in
#                              flight per thread cost
# The three run at FP16 only, one after the other on each matrix, so that they see the same machine.
#
#   source experiments/slurm/env.sh
#   experiments/slurm/build.sh            # builds the two control binaries next to the campaign one
#   sbatch experiments/slurm/fp16_controls.sbatch
#
# Env:  SPMV_LIST      matrix list (default experiments/lists/all.txt, the 74 of the campaign)
# Output: ${SPMV_RESULTS_DIR}/fp16_controls/<binary>/csr_layouts_rcm.csv, .log and _tpr.csv, the same
#         files and protocol as the campaign, RCM ordering only.
#
#SBATCH --job-name=spmv-fp16ctl
# the partition comes from SBATCH_PARTITION, which env.sh exports from SPMV_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=spmv-fp16ctl-%j.out

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f "${ROOT}/experiments/slurm/env.sh" ]] || { echo "submit from the repository root, not from ${ROOT}"; exit 1; }
# shellcheck disable=SC1091
source "${ROOT}/experiments/slurm/env.sh"
spmv_load_modules

LIST="${SPMV_LIST:-${ROOT}/experiments/lists/all.txt}"
BINARIES=(bench_csr_width bench_csr_width_fp16acc bench_csr_width_r16unroll)
OUT_ROOT="${SPMV_RESULTS_DIR}/fp16_controls"

echo "=== job ${SLURM_JOB_ID:-none} on $(hostname), $(date '+%F %T') ==="
spmv_describe
echo "list         ${LIST}"
echo "protocol     reps ${REPS:-20}, warmup ${WARMUP:-3}, min sample ${MIN_SAMPLE_MS:-10} ms, FP16 only"
for binary in "${BINARIES[@]}"; do
  bench="${SPMV_BUILD_DIR}/${binary}"
  [[ -x "${bench}" ]] || { echo "no ${bench}: run experiments/slurm/build.sh"; exit 1; }
  missing="$(ldd -r "${bench}" 2>&1 | grep -E 'undefined symbol|not found' || true)"
  [[ -z "${missing}" ]] || { echo "${bench} cannot be loaded in this job:"; echo "${missing}"; exit 1; }
  mkdir -p "${OUT_ROOT}/${binary}"
done
"${SPMV_BUILD_DIR}/device_report" > "${OUT_ROOT}/device.txt"

# One matrix at a time through the three binaries, so a drift of the machine during the job affects
# the three alike. run_campaign.sh skips a matrix its log already marks as done, so the job resumes.
while IFS= read -r entry; do
  [[ -z "${entry}" || "${entry}" == \#* ]] && continue
  printf '%s\n' "${entry}" > "${OUT_ROOT}/current.txt"
  for binary in "${BINARIES[@]}"; do
    OUT="${OUT_ROOT}/${binary}" BENCH="${SPMV_BUILD_DIR}/${binary}" MATRIX_ROOT="${MATRIX_ROOT}" ORDERING="rcm" \
    PRECISION="fp16" REPS="${REPS:-20}" WARMUP="${WARMUP:-3}" MIN_SAMPLE_MS="${MIN_SAMPLE_MS:-10}" \
    SPMV_FORCE_RESUME=1 "${ROOT}/experiments/run_campaign.sh" "${OUT_ROOT}/current.txt"
  done
done < "${LIST}"
rm -f "${OUT_ROOT}/current.txt"
echo "=== done $(date '+%F %T') ==="
