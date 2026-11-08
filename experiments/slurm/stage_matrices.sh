#!/usr/bin/env bash
# stage_matrices.sh — put the 74 SuiteSparse matrices on the parallel filesystem.
#
# About 70 GB uncompressed, so not in $HOME. Compute nodes have no route to the outside, and the
# login nodes cut off long processes, so the download belongs on the serial partition: --submit
# sends this same script there. It is resumable, a matrix already on disk is not fetched again,
# and a job that runs out of walltime is continued by submitting it again.
#
# Usage:  source experiments/slurm/env.sh
#         experiments/slurm/stage_matrices.sh [--submit] [--tiers S,M,L,XL] [--jobs 4] [--check]
# Env:    MATRIX_ROOT (default $SPMV_DATA_ROOT/matrices), SPMV_SERIAL_PARTITION
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/experiments/slurm/env.sh"
TIERS="S,M,L,XL"
JOBS=4
CHECK_ONLY=0
SUBMIT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit) SUBMIT=1; shift ;;
    --tiers)  TIERS="$2"; shift 2 ;;
    --jobs)   JOBS="$2"; shift 2 ;;
    --check)  CHECK_ONLY=1; shift ;;
    *) echo "stage_matrices.sh: unknown option $1"; exit 1 ;;
  esac
done

if [[ ${SUBMIT} -eq 1 ]]; then
  exec sbatch --job-name=spmv-stage --partition="${SPMV_SERIAL_PARTITION}" \
              --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=04:00:00 \
              --output="spmv-stage-%j.out" \
              --wrap "${ROOT}/experiments/slurm/stage_matrices.sh --tiers ${TIERS} --jobs ${JOBS}"
fi

mkdir -p "${MATRIX_ROOT}"
# Lustre: the largest matrices are several GB of text read start to end, and a single stripe caps
# them at the bandwidth of one object storage target. Harmless where lfs does not exist.
command -v lfs >/dev/null 2>&1 && lfs setstripe -c 4 "${MATRIX_ROOT}" 2>/dev/null || true

verify() {
  local staged=0 missing=0 corrupt=0 bytes=0 entry file
  while IFS= read -r entry; do
    [[ -z "${entry}" ]] && continue
    file="${MATRIX_ROOT}/${entry#matrices/}"
    if [[ ! -f "${file}" ]]; then
      missing=$((missing + 1))
      echo "  [missing]   ${entry#matrices/}"
    elif [[ "$(head -c 14 "${file}")" != "%%MatrixMarket" ]]; then
      corrupt=$((corrupt + 1))
      echo "  [corrupt]   ${file}"
    else
      staged=$((staged + 1))
      bytes=$((bytes + $(stat -c %s "${file}")))
    fi
  done < "${ROOT}/experiments/lists/all.txt"
  echo "  ${staged} complete, ${missing} missing, ${corrupt} not Matrix Market, $((bytes / 1073741824)) GiB"
  return $((missing + corrupt > 0))
}

if [[ ${CHECK_ONLY} -eq 1 ]]; then
  echo "=== checking ${MATRIX_ROOT} ==="
  verify
  exit $?
fi

echo "=== staging tiers ${TIERS} into ${MATRIX_ROOT} ==="
python3 -c "import urllib.request; urllib.request.urlopen('https://sparse.tamu.edu', timeout=30)" 2>/dev/null \
  || { echo "no route to sparse.tamu.edu from $(hostname): run this on a login node or on ${SPMV_SERIAL_PARTITION}"; exit 1; }

date '+start %F %T'
python3 "${ROOT}/matrices/download.py" --root "${MATRIX_ROOT}" --tiers "${TIERS}" --jobs "${JOBS}"
date '+end   %F %T'

echo "=== checking ${MATRIX_ROOT} ==="
if verify; then
  echo "dataset complete."
else
  echo "incomplete: run this script again (it skips what is already there)."
  exit 1
fi
