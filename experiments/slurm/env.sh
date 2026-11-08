#!/usr/bin/env bash
# env.sh — every site-specific setting of the campaign, in one place.
#
# Source it before submitting: it exports the SBATCH_* variables that carry the account, the
# partition and the QoS into every job, so the job scripts hold no site-specific identifiers.
#
#   source experiments/slurm/env.sh          # from the repository root
#   sbatch experiments/slurm/campaign.sbatch
#
# Every variable can be set beforehand and is then left alone:
#
#   SPMV_ACCOUNT    Slurm account to charge          (required to submit; check with your site)
#   SPMV_DATA_ROOT  where matrices are staged        (default $SCRATCH/spmv-index-width)
#   SPMV_PARTITION  GPU partition                    (default gpu)
#   SPMV_QOS        QoS, empty for the default one   (debug for a 30-minute test)
#   SPMV_SERIAL_PARTITION  partition for the download and the build (default serial)
#   SPMV_CUDA_MODULE       CUDA module, empty to pick the newest available
#   SPMV_EXTRA_MODULES     further modules, e.g. "cmake gcc/12.2.0"
#   SPMV_BUILD_DIR         build directory           (default <repo>/build-a100)
#   SPMV_RESULTS_DIR       output directory          (default <repo>/data/results/a100)
#
# This file is sourced, not executed, so it must not call exit and must not set -e.

SPMV_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export SPMV_REPO_ROOT

# --- where the data lives ------------------------------------------------------------------------
# The 74 matrices need about 70 GB, which does not fit the $HOME quota: stage them on $SCRATCH
# (no quota, cleaned after 40 days) or on $WORK (kept, shared by the account).
export SPMV_DATA_ROOT="${SPMV_DATA_ROOT:-${SCRATCH:-${WORK:-$HOME}}/spmv-index-width}"
export MATRIX_ROOT="${MATRIX_ROOT:-${SPMV_DATA_ROOT}/matrices}"
# The build and results directories of another repository survive in a shell that sourced its
# env.sh first, since both repositories use these names. A value outside this repository is that
# leftover, not a choice, so it is replaced, and said so.
for spmv_variable in SPMV_BUILD_DIR SPMV_RESULTS_DIR; do
  if [[ -n "${!spmv_variable:-}" && "${!spmv_variable}" != "${SPMV_REPO_ROOT}"/* ]]; then
    echo "env.sh: ${spmv_variable}=${!spmv_variable} belongs to another repository, reset for ${SPMV_REPO_ROOT}" >&2
    unset "${spmv_variable}"
  fi
done
unset spmv_variable
export SPMV_BUILD_DIR="${SPMV_BUILD_DIR:-${SPMV_REPO_ROOT}/build-a100}"
export SPMV_RESULTS_DIR="${SPMV_RESULTS_DIR:-${SPMV_REPO_ROOT}/data/results/a100}"

# --- Slurm ----------------------------------------------------------------------------------------
export SPMV_PARTITION="${SPMV_PARTITION:-gpu}"
export SPMV_SERIAL_PARTITION="${SPMV_SERIAL_PARTITION:-serial}"
# Without an account sbatch falls back to the default one of the user, which on a machine where
# several projects coexist is usually the expired one, and the submission fails with a message that
# does not name the cause. Saying it here costs nothing and saves a round trip.
if [[ -n "${SPMV_ACCOUNT:-}" ]]; then
  export SBATCH_ACCOUNT="${SPMV_ACCOUNT}"
else
  echo "env.sh: SPMV_ACCOUNT is not set, so sbatch will use your default account." >&2
  echo "        your site lists the ones you can charge; export SPMV_ACCOUNT and source again." >&2
fi
[[ -n "${SPMV_QOS:-}" ]] && export SBATCH_QOS="${SPMV_QOS}"
# The partition travels the same way. sbatch reads SBATCH_PARTITION and an environment variable beats
# a #SBATCH directive inside the script, which is what lets one set of job files serve more than one
# partition of the same machine. Running on a different GPU means three variables and nothing else:
#
#   export SPMV_PARTITION=<the other partition>
#   export SPMV_ARCH=<its compute capability without the dot, 80 for A100, 90 for Hopper>
#   export SPMV_BUILD_DIR=<repo>/build-<that gpu>   SPMV_RESULTS_DIR=<repo>/results/<that gpu>
#
# The build directory and the results directory have to move as well, or the second GPU overwrites
# the binaries and the measurements of the first, and the device constants of the model would then
# belong to one machine while the times belong to another.
export SBATCH_PARTITION="${SPMV_PARTITION}"

# --- modules ----------------------------------------------------------------------------------------
# Module names drift between system updates, so nothing is hard-wired: the newest CUDA the machine
# offers is taken unless SPMV_CUDA_MODULE names one. Run experiments/slurm/preflight.sh to see
# what was picked before submitting anything.

# Makes "module" usable in a non-interactive shell, where the login profile may not have run.
spmv_init_modules() {
  if ! type module >/dev/null 2>&1; then
    # shellcheck disable=SC1090,SC1091
    for init in /usr/share/lmod/lmod/init/bash "${MODULESHOME:-/usr/share/Modules}/init/bash"; do
      [[ -r "${init}" ]] && source "${init}" && break
    done
  fi
  type module >/dev/null 2>&1
}

# Loads the first candidate that exists and echoes it; used for modules whose version varies.
spmv_load_first() {
  local candidate
  for candidate in "$@"; do
    if module load "${candidate}" >/dev/null 2>&1; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

# The release nvcc actually reports, which is the only toolkit version that matters.
spmv_nvcc_version() {
  nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p'
}

# Loads the first CUDA module that really puts its own nvcc in PATH. A module can load with status
# 0 and leave nvcc alone (a stale name, a module that only sets library paths), and the build would
# then silently use whatever toolkit was already there; that is worse than not loading anything.
spmv_load_cuda() {
  local candidate wanted effective
  for candidate in "$@"; do
    module load "${candidate}" >/dev/null 2>&1 || continue
    hash -r 2>/dev/null || true   # the shell caches where nvcc was, and the module just moved it
    effective="$(spmv_nvcc_version)"
    wanted="${candidate#cuda/}"
    if [[ -n "${effective}" ]] && { [[ "${candidate}" == "cuda" ]] || [[ "${effective}" == "${wanted}"* ]]; }; then
      export SPMV_CUDA_MODULE="${candidate}"
      export SPMV_CUDA_VERSION="${effective}"
      return 0
    fi
    module unload "${candidate}" >/dev/null 2>&1 || true
    hash -r 2>/dev/null || true
  done
  return 1
}

# Loads the toolchain. Returns non-zero if CUDA could not be loaded, which every caller treats as
# fatal: building or running against the wrong toolkit is worse than not running.
spmv_load_modules() {
  local extra
  if ! spmv_init_modules; then
    echo "env.sh: no module command in this shell" >&2
    return 1
  fi
  module load profile/base >/dev/null 2>&1 || true
  if [[ -n "${SPMV_CUDA_MODULE:-}" ]]; then
    spmv_load_cuda "${SPMV_CUDA_MODULE}" || {
      echo "env.sh: ${SPMV_CUDA_MODULE} does not load or does not provide its own nvcc" \
           "(nvcc reports '$(spmv_nvcc_version)'); 'module avail cuda' lists what exists" >&2
      return 1
    }
  else
    spmv_load_cuda cuda/12.6 cuda/12.5 cuda/12.4 cuda/12.3 cuda/12.2 cuda/12.1 cuda || {
      echo "env.sh: no CUDA module loaded; set SPMV_CUDA_MODULE from 'module avail cuda'" >&2
      return 1
    }
  fi
  # cmake >= 3.20 is required and the system one is often older or absent.
  spmv_load_first cmake cmake/3.27.7 >/dev/null 2>&1 || true
  # The host compiler is part of the result, so it is chosen here rather than left to whatever the
  # login shell happens to carry. The system g++ of RHEL 8 is 8.5, which predates the C++17 library
  # features this code uses in places, and a build that silently picks it is not the reference build.
  spmv_load_first gcc gcc/12.2.0 >/dev/null 2>&1 || \
    echo "env.sh: no gcc module loaded, the system compiler ($(g++ -dumpversion 2>/dev/null)) will be used" >&2
  for extra in ${SPMV_EXTRA_MODULES:-}; do
    module load "${extra}" >/dev/null 2>&1 || echo "env.sh: cannot load ${extra}, continuing" >&2
  done
  return 0
}

# One line describing the environment, for the head of every log.
spmv_describe() {
  echo "repo         ${SPMV_REPO_ROOT}"
  echo "matrices     ${MATRIX_ROOT}"
  echo "build        ${SPMV_BUILD_DIR}"
  echo "results      ${SPMV_RESULTS_DIR}"
  echo "cuda module  ${SPMV_CUDA_MODULE:-<not loaded>} (nvcc ${SPMV_CUDA_VERSION:-$(spmv_nvcc_version)})"
  echo "account      ${SBATCH_ACCOUNT:-<unset: pass --account to sbatch>}"
  echo "partition    ${SPMV_PARTITION} (serial: ${SPMV_SERIAL_PARTITION})"
}
