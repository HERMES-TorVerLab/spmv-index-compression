#!/bin/bash
# k_study.sbatch - what K is made of, inside the model domain and at each of its limits.
#
#   source experiments/slurm/env.sh
#   sbatch experiments/slurm/k_study.sbatch
#
# The model writes the speedup as the byte ratio times K/K_ref, with K = Q/t the rate at which a run
# retires its counted bytes. With the DRAM traffic Q_real from the counters, K splits exactly into
#
#   K = B * (Q / Q_real) * beta,      beta = Q_real / (t B)
#
# the counted share of the real traffic times the share of the bandwidth the kernel keeps busy.
# Inside the domain Q/Q_real is one within a few per cent, so K/K_ref is the ratio of the two betas,
# and beta falls with the bytes each load instruction carries. This job measures both factors on the
# matrices of experiments/lists/k_study.txt, nine inside the domain and one group per limit of the
# model (L2-resident, imbalanced, short rows with a large x), with the counters that explain beta:
#
#   sm__warps_active                     achieved occupancy, the warps that can keep loads in flight
#   smsp__inst_executed_op_global_ld     global load instructions, so bytes per load are measured
#   l1tex__t_requests / t_sectors (ld)   sectors per request, the coalescing of the loads
#   lts__t_sectors_srcunit_tex_op_read   L2 traffic, and its hit rate, for the L2-resident matrices
#   dram__throughput                     beta as the tool reports it, a check on Q_real / (t B)
#   sm__cycles_active avg and max        the spread of work over the SMs, for the imbalanced ones
#
# The metrics do not fit one replay pass, so each kernel is replayed a few times, hence the long time
# limit. Every matrix leaves <name>.ncu-rep, the report
# Nsight Compute opens (ncu-ui <name>.ncu-rep), <name>.ncu.csv, the counters exported from it, and
# <name>.log, what the benchmark printed. The report must be opened with a Nsight Compute at least as
# recent as the ncu that wrote it; ncu --version on the login node tells which.
#
# Env:  SPMV_LIST      matrices to profile   (default experiments/lists/k_study.txt)
#       SPMV_NCU_KERNELS  which kernels to profile, as an ncu --kernel-name filter
#                      (default the three layouts of the paper, regex:k_csr_(cumulative|row_relative|lane_packed));
#                      SPMV_NCU_KERNELS=regex:. profiles every kernel, cuSPARSE included
#       SPMV_STUDY     name of the output directory under the results (default k_study_rcm)
#       SPMV_BENCH     binary to profile, a control build of CMakeLists.txt (default bench_csr_width)
#       SPMV_NCU_SET   a section set of Nsight Compute to collect on top of the metrics, such as
#                      "full", which fills every page of the report (memory charts, stalls per
#                      instruction) at about forty replays a kernel; empty by default
#
# With SPMV_NCU_SET=full a large matrix alone can take most of a day, so the job is meant to run as an
# array, one matrix per task, each with its own time limit:
#
#   SPMV_NCU_SET=full sbatch --array=0-18 --time=24:00:00 experiments/slurm/k_study.sbatch
#
# A task profiles only the matrix whose position in the list, comments left out, is its task id.
# Without --array the job walks the whole list, as before.
#
#SBATCH --job-name=spmv-kstudy
# the partition comes from SBATCH_PARTITION, which env.sh exports from SPMV_PARTITION,
# so one job file serves every partition of the machine
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=180G
#SBATCH --time=10:00:00
#SBATCH --output=spmv-kstudy-%A_%a.out

set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f "${ROOT}/experiments/slurm/env.sh" ]] || { echo "submit from the repository root, not from ${ROOT}"; exit 1; }
# shellcheck disable=SC1091
source "${ROOT}/experiments/slurm/env.sh"
spmv_load_modules

LIST="${SPMV_LIST:-${ROOT}/experiments/lists/k_study.txt}"
BENCH="${SPMV_BUILD_DIR}/${SPMV_BENCH:-bench_csr_width}"
NCU_SET="${SPMV_NCU_SET:-}"
KERNELS="${SPMV_NCU_KERNELS:-regex:k_csr_(cumulative|row_relative|lane_packed)}"
OUT="${SPMV_RESULTS_DIR}/${SPMV_STUDY:-k_study_rcm}${NCU_SET:+_${NCU_SET}}"
SET_FLAGS=()
[[ -n "${NCU_SET}" ]] && SET_FLAGS=(--set "${NCU_SET}")
TASK="${SLURM_ARRAY_TASK_ID:-}"

echo "=== job ${SLURM_JOB_ID:-none} on $(hostname), $(date '+%F %T') ==="
spmv_describe
echo "list         ${LIST}"
[[ -x "${BENCH}" ]] || { echo "no ${BENCH}: run experiments/slurm/build.sh"; exit 1; }
command -v ncu >/dev/null 2>&1 || { echo "no ncu in PATH: it ships with the CUDA toolkit, so 'module load cuda' should bring it"; exit 1; }

# The binary links cuSPARSE and a batch job does not always get the module environment of the build.
missing="$(ldd -r "${BENCH}" 2>&1 | grep -E 'undefined symbol|not found' || true)"
if [[ -n "${missing}" ]]; then
  echo "${BENCH} cannot be loaded in this job:"
  echo "${missing}" | sed 's/^/  /'
  exit 1
fi
mkdir -p "${OUT}"

# The matrices are reordered with reverse Cuthill-McKee, as in the campaign the paper reports, so the
# counters describe the same access to x and the same packed widths as the timed runs. The permutation
# is read from the .rcm.perm cached next to each matrix by that campaign.
# One launch per configuration. The timing is meaningless at this protocol and is not used: every
# number this job produces comes from the counters, and the times come from the campaign. The rows
# the benchmark writes go to <name>.times.csv beside the report, so a profiled run never reaches the
# campaign file results/csr_width.csv.
PROTOCOL=(--ordering rcm --reps 1 --warmup 0 --min-sample-ms 0)

# The four stall reasons are shares of the cycles a warp was active, so they are comparable across
# kernels of different length without any normalisation. The set needs a few replay passes.
METRICS=smsp__warp_issue_stalled_wait_per_warp_active.pct,\
smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_no_instruction_per_warp_active.pct,\
smsp__inst_executed.sum,\
dram__bytes_read.sum,\
gpu__time_duration.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__inst_executed_op_global_ld.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
lts__t_sectors_srcunit_tex_op_read.sum,\
lts__t_sector_op_read_hit_rate.pct,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__cycles_active.avg,\
sm__cycles_active.max

position=-1
while IFS= read -r entry; do
  [[ -z "${entry}" || "${entry}" == \#* ]] && continue
  position=$((position + 1))
  [[ -n "${TASK}" && "${position}" -ne "${TASK}" ]] && continue
  matrix_file="${MATRIX_ROOT}/${entry#matrices/}"
  name="$(basename "${matrix_file}" .mtx)"
  [[ -f "${matrix_file}" ]] || { echo "[missing] ${matrix_file}"; continue; }
  echo
  echo "=== ${name} === $(date '+%T')"
  # The caches are not flushed between replays, so a matrix that fits in the L2 is measured warm, as
  # in the campaign, where every product follows another; above the L2 the flag changes nothing.
  # Only the three kernels the question is about. cuSPARSE is skipped because its kernel is not ours
  # to explain, and the length-encoded variants are not in the paper.
  # The report is kept, so the kernels can be opened in Nsight Compute, and the table the analysis
  # reads is exported from it. LC_ALL=C writes the numbers with a decimal point whatever the locale
  # of the job, which spares analyse_stalls.py the guess between the two separators.
  ncu --target-processes all \
      --kernel-name "${KERNELS}" \
      --kernel-name-base demangled \
      --cache-control none \
      ${SET_FLAGS[@]+"${SET_FLAGS[@]}"} \
      --metrics "${METRICS}" \
      --force-overwrite \
      --export "${OUT}/${name}" \
      "${BENCH}" --matrix "${matrix_file}" "${PROTOCOL[@]}" --csv "${OUT}/${name}.times.csv" > "${OUT}/${name}.log" 2>&1
  status=$?
  if [[ ${status} -ne 0 || ! -f "${OUT}/${name}.ncu-rep" ]]; then
    echo "  ncu exited ${status}, see ${OUT}/${name}.log"
    tail -5 "${OUT}/${name}.log" | sed 's/^/    /'
    continue
  fi
  LC_ALL=C ncu --import "${OUT}/${name}.ncu-rep" --csv --page raw --print-units base > "${OUT}/${name}.ncu.csv"
  echo "  report $(du -h "${OUT}/${name}.ncu-rep" | cut -f1), $(grep -c . "${OUT}/${name}.ncu.csv") lines of counters"
done < "${LIST}"

echo
echo "reports and counters in ${OUT}"
ls -la "${OUT}"
echo
echo "copy the directory off, next to data/results/a100/stalls"
