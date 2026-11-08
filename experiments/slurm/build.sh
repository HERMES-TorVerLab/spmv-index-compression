#!/usr/bin/env bash
# build.sh — build the benchmark for the A100 (compute capability 8.0) with the system CUDA module.
#
# CMAKE_CUDA_ARCHITECTURES is set explicitly rather than left at "native": a login node has no GPU,
# so "native" would either fail or build for whatever the compile node happens to be.
#
# Usage:  source experiments/slurm/env.sh
#         experiments/slurm/build.sh              # here (a few minutes of CPU)
#         experiments/slurm/build.sh --submit     # as a job, if the login-node limits kill it
# Env:    SPMV_BUILD_DIR, SPMV_CUDA_MODULE, SPMV_ARCH (default 80), SPMV_JOBS (default 8)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/experiments/slurm/env.sh"
ARCH="${SPMV_ARCH:-80}"
JOBS="${SPMV_JOBS:-8}"

if [[ "${1:-}" == "--submit" ]]; then
  exec sbatch --job-name=spmv-build --partition="${SPMV_SERIAL_PARTITION}" \
              --nodes=1 --ntasks=1 --cpus-per-task="${JOBS}" --mem=16G --time=01:00:00 \
              --output="spmv-build-%j.out" --wrap "${ROOT}/experiments/slurm/build.sh"
fi

spmv_load_modules
spmv_describe
echo "architecture ${ARCH} (A100 = 80)"

cmake -B "${SPMV_BUILD_DIR}" -S "${ROOT}" \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${ARCH}"
cmake --build "${SPMV_BUILD_DIR}" -j "${JOBS}"

for binary in bench_csr_width device_report characterize_matrix; do
  [[ -x "${SPMV_BUILD_DIR}/${binary}" ]] || { echo "build.sh: ${binary} missing"; exit 1; }
done
echo
echo "built in ${SPMV_BUILD_DIR}:"
ls -la "${SPMV_BUILD_DIR}"/bench_csr_width "${SPMV_BUILD_DIR}"/device_report \
       "${SPMV_BUILD_DIR}"/characterize_matrix
echo "the binaries need a GPU to run: submit experiments/slurm/campaign.sbatch"
