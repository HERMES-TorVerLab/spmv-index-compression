#!/usr/bin/env bash
# characterize_all.sh — run characterize_matrix over every downloaded matrix.
#
# One CSV per matrix in matrices/characterization/, skipped if already present.
# Small files run in parallel; files above LARGE_BYTES run one at a time,
# because the reader holds the text, the triplets and the CSR in RAM together.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL="${TOOL:-${HERE}/../build/characterize_matrix}"
# MATRIX_ROOT: directory holding <category>/<name>/<name>.mtx (default: this directory).
MATRIX_ROOT="${MATRIX_ROOT:-${HERE}}"
# ORDERING=rcm characterizes P A P^T into characterization_rcm/ and caches the
# permutation next to each matrix as <name>.rcm.perm (reused by bench_csr_width).
ORDERING="${ORDERING:-original}"
# OUT: where the per-matrix CSVs go. The default follows the ordering, and an explicit value is for
# keeping a pass apart, for instance while a new column of the tool is being validated against the
# committed reference.
if [[ -z "${OUT:-}" ]]; then
  if [[ "${ORDERING}" == "original" ]]; then OUT="${HERE}/characterization"; else OUT="${HERE}/characterization_${ORDERING}"; fi
fi
JOBS="${JOBS:-16}"
LARGE_BYTES="${LARGE_BYTES:-4000000000}"

mkdir -p "${OUT}"
[[ -x "${TOOL}" ]] || { echo "build ${TOOL} first"; exit 1; }

characterize_one() {
  local matrix_path="$1" name
  name="$(basename "${matrix_path}" .mtx)"
  [[ -s "${OUT}/${name}.csv" ]] && return 0
  if "${TOOL}" "${matrix_path}" "${OUT}/${name}.csv.partial" --ordering "${ORDERING}" 2>>"${OUT}/errors.log"; then
    mv "${OUT}/${name}.csv.partial" "${OUT}/${name}.csv"
    echo "[ok] ${name}"
  else
    rm -f "${OUT}/${name}.csv.partial"
    echo "[fail] ${name} (see errors.log)"
  fi
}
export -f characterize_one
export TOOL OUT ORDERING

# -r: run nothing when find matches nothing (GNU xargs would otherwise call the
# command once with no argument).
find "${MATRIX_ROOT}" -mindepth 3 -maxdepth 3 -name '*.mtx' -size -"${LARGE_BYTES}"c -print0 \
  | xargs -0 -r -n1 -P "${JOBS}" bash -c 'characterize_one "$0"'
find "${MATRIX_ROOT}" -mindepth 3 -maxdepth 3 -name '*.mtx' -size +"$((LARGE_BYTES - 1))"c -print0 \
  | xargs -0 -r -n1 -P 1 bash -c 'characterize_one "$0"'
