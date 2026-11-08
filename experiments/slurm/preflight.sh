#!/usr/bin/env bash
# preflight.sh — what this machine actually offers, before anything is submitted.
#
# Runs on a login node in a few seconds and checks nothing but facts: the module that will be
# loaded, the toolchain versions, the filesystem the matrices will go to, the Slurm account and
# how much of the dataset is already staged. Nothing here costs budget.
#
# Usage:  source experiments/slurm/env.sh && experiments/slurm/preflight.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/experiments/slurm/env.sh"

problems=0
note() { printf '  %-14s %s\n' "$1" "$2"; }
bad()  { printf '  %-14s %s\n' "$1" "$2"; problems=$((problems + 1)); }

echo "=== host ==="
note "hostname" "$(hostname)"
note "os" "$(source /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-unknown}")"

echo "=== modules ==="
if spmv_load_modules; then
  note "cuda module" "${SPMV_CUDA_MODULE}"
  note "nvcc" "$(nvcc --version 2>/dev/null | tail -1)"
else
  bad "cuda module" "not loaded (see 'module avail cuda')"
fi
if command -v cmake >/dev/null 2>&1; then
  cmake_version="$(cmake --version | head -1 | awk '{print $3}')"
  if printf '%s\n3.20\n' "${cmake_version}" | sort -V | head -1 | grep -qx "3.20"; then
    note "cmake" "${cmake_version}"
  else
    bad "cmake" "${cmake_version} is older than the required 3.20 (module load cmake)"
  fi
else
  bad "cmake" "not found (module load cmake)"
fi
note "host compiler" "$(${CXX:-g++} --version 2>/dev/null | head -1)"
# 3.6 is enough: the analysis avoids what arrived later (subprocess capture_output, text=, and
# statistics.quantiles, which it carries its own copy of). Anything older will not run it.
python_ok="$(python3 -c 'import sys; print("yes" if sys.version_info >= (3, 6) else "no")' 2>/dev/null)"
if [[ "${python_ok}" == "yes" ]]; then
  note "python3" "$(python3 --version 2>&1) (the analysis runs here)"
else
  bad "python3" "$(python3 --version 2>&1), the analysis needs 3.6 or newer"
fi
python3 -c "import matplotlib" 2>/dev/null \
  && note "matplotlib" "present (the analysis can run here)" \
  || note "matplotlib" "absent (run the analysis elsewhere, or pip install --user matplotlib)"

echo "=== storage ==="
for name in HOME WORK SCRATCH FAST; do
  path="${!name:-}"
  [[ -n "${path}" && -d "${path}" ]] && note "\$${name}" "${path}" || note "\$${name}" "<unset>"
done
note "data root" "${SPMV_DATA_ROOT}"
case "${SPMV_DATA_ROOT}" in
  "${HOME}"/*) bad "data root" "is under \$HOME, whose quota is far below the 70 GB of matrices" ;;
esac
mkdir -p "${SPMV_DATA_ROOT}" 2>/dev/null \
  || bad "data root" "cannot be created"
note "free space" "$(df -h "${SPMV_DATA_ROOT}" 2>/dev/null | awk 'NR==2 {print $4" available on "$1}')"
# cindata walks the filesystem to report the quota and can take minutes on a busy day, while this
# script is meant to last seconds. It is informative only, so it is given a short deadline and its
# absence is not a failure.
if command -v cindata >/dev/null 2>&1; then
  echo "  --- cindata ---"
  if ! timeout 20 cindata 2>&1 | sed 's/^/  /'; then
    echo "  (cindata did not answer within 20s, skipped)"
  fi
fi

echo "=== dataset ==="
staged=0
missing=0
bytes=0
while IFS= read -r entry; do
  [[ -z "${entry}" ]] && continue
  file="${MATRIX_ROOT}/${entry#matrices/}"
  if [[ -f "${file}" ]]; then
    staged=$((staged + 1))
    bytes=$((bytes + $(stat -c %s "${file}")))
  else
    missing=$((missing + 1))
  fi
done < "${ROOT}/experiments/lists/all.txt"
note "staged" "${staged} of $((staged + missing)) matrices, $((bytes / 1073741824)) GiB in ${MATRIX_ROOT}"
[[ ${missing} -gt 0 ]] && note "next" "experiments/slurm/stage_matrices.sh (or sbatch it)"

echo "=== slurm ==="
if command -v sinfo >/dev/null 2>&1; then
  note "account" "${SBATCH_ACCOUNT:-<unset>: set SPMV_ACCOUNT and source env.sh again}"
  [[ -z "${SBATCH_ACCOUNT:-}" ]] && problems=$((problems + 1))
  for partition in "${SPMV_PARTITION}" "${SPMV_SERIAL_PARTITION}"; do
    if sinfo -h -p "${partition}" -o "%P %a %l %D" 2>/dev/null | grep -q .; then
      note "${partition}" "$(sinfo -h -p "${partition}" -o '%a, max walltime %l, %D nodes' | head -1)"
    else
      bad "${partition}" "unknown partition on this cluster (see 'sinfo -s')"
    fi
  done
else
  bad "slurm" "no sinfo: this is not a the cluster login node"
fi

echo
if [[ ${problems} -eq 0 ]]; then
  echo "preflight clean. Next:"
  echo "  experiments/slurm/build.sh                      # or --submit on the serial partition"
  echo "  experiments/slurm/stage_matrices.sh --submit    # about 70 GB, resumable"
  echo "  sbatch experiments/slurm/campaign.sbatch        # the paper campaign, one A100"
else
  echo "${problems} problem(s) above; fix them before submitting."
fi
exit $((problems > 0))
