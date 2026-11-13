# spmv-index-compression

Code and data of the paper *Turning Index Compression into Speedup in SpMV Kernels on GPUs*.

One CUDA program measures CSR SpMV on 74 SuiteSparse matrices at FP64, FP32 and FP16 with several ways
of storing the column indices, against 32-bit CSR and cuSPARSE.

| layout | column index | exact when |
|---|---|---|
| CSR-i32 | 32-bit absolute, the reference | always |
| CSR-i16 | 16-bit absolute | `n` is at most 2^16 |
| CSR-r16 | 16-bit offset from the first column of the row | every row spans fewer than 2^16 columns |
| CSR-lp | bit-packed differences, one chain per lane of the row | always |
| cuSPARSE | `cusparseSpMV`, CSR_ALG1 and CSR_ALG2, FP16 computed in FP32 | always |

Every layout runs one vector of `T` threads per row, `T` in {2, 4, 8, 16, 32}, and keeps the value array
and the order of the nonzeros of CSR-i32. At the same `T` its result must be bit-identical to CSR-i32,
and the benchmark checks this on every matrix.

## Reproducing, from the cheapest level to the dearest

| level | what it does | needs | time |
|---|---|---|---|
| 1 | regenerates the Results section and the figures from `data/results` | Python | about a minute |
| 2 | builds the code, measures two small matrices and checks them | one NVIDIA GPU, CUDA | minutes |
| 3 | repeats the campaign on all 74 matrices | one GPU, about 70 GB of disk | hours |
| 4 | repeats the hardware counters on 19 matrices | access to the GPU performance counters | hours |

### Level 1, the paper from the published measurements

```bash
pip install -r requirements.txt
analysis/reproduce_paper.sh
python3 analysis/make_layout_figure.py --output-dir output/paper
```

The output is in `output/paper`, `sections/results.tex` and `figures/*.pdf`.

### Level 2, a quick check on your GPU

```bash
experiments/smoke.sh
```

It builds the benchmark, unpacks `dawson5` and `rim` from `data/smoke_matrices`, verifies their checksums, measures them and runs
`analysis/compare_campaign.py`, which exits with an error if a result differs from CSR-i32 or a byte count
differs from the published one. The five samples of this run are too few for its speedups to be compared
with the paper.

### Level 3, the campaign

```bash
# Unpack matrices_*.tar.gz and rcm_permutations.tar.gz of the data record (https://doi.org/10.5281/zenodo.24060361) in matrices/.
# About 70 GB. matrices/download.py fetches from the SuiteSparse Collection, if its files are reachable.
(cd matrices && sha256sum -c ../data/matrices.sha256)
cmake -B build -S . && cmake --build build -j 8
build/device_report > results/local/device.txt    # create results/local first; takes a few minutes
experiments/run_campaign.sh                       # writes results/local/csr_layouts_rcm.csv, resumes if killed
python3 analysis/compare_campaign.py results/local/csr_layouts_rcm.csv
```

The first run computes each reverse Cuthill-McKee permutation and caches it beside the matrix as
`<name>.rcm.perm`, and later runs reuse it. The published campaign in the same ordering took about 2 hours
on the A100 and 6 hours on the Quadro RTX 5000, on top of the permutations. Keep the defaults of the
protocol, 20 samples of at least 10 ms after 3 warm-up products, or the campaigns stop being comparable.

A matrix that does not fit the memory of the GPU is skipped and reported. The largest ones need the 64 GB
of the A100 used in the paper.

On a cluster, `experiments/slurm` holds the same steps as Slurm jobs. Set `SPMV_ACCOUNT`, `SPMV_PARTITION`
and `SPMV_CUDA_MODULE` for your site, then `source experiments/slurm/env.sh`, run
`experiments/slurm/build.sh` and `sbatch experiments/slurm/campaign.sbatch`.

### Level 4, the hardware counters

```bash
source experiments/slurm/env.sh
sbatch experiments/slurm/k_study.sbatch           # one <name>.ncu.csv per matrix of experiments/lists/k_study.txt
sbatch experiments/slurm/r16_loads.sbatch         # CSR-r16 beside its three bytes-per-load controls
```

Outside Slurm the same `ncu` command is in that file. It needs access to the performance counters of the
GPU, which on a workstation usually means running as root.

## From the data to each figure and table

Everything in this table is produced by level 1, except the last two rows, which have their own command.

| in the paper | file | script | data |
|---|---|---|---|
| Fig. Roofline | `figures/roofline.pdf` | `analysis/make_results.py` | `a100/csr_layouts_rcm*.csv`, `a100/device.txt` |
| Fig. speedup | `figures/speedup.pdf` | `analysis/make_results.py` | `a100/csr_layouts_rcm*.csv` |
| Fig. performance | `figures/performance.pdf` | `analysis/make_results.py` | `a100/csr_layouts_rcm*.csv`, `a100/cusparse_rcm` |
| Fig. three factors | `figures/limits_brief.pdf`, `figures/limits.pdf` | `analysis/make_results.py` | `a100/csr_layouts_rcm*.csv`, `a100/k_study_rcm` |
| Fig. prediction, extended version | `figures/prediction.pdf` | `analysis/make_results.py` | `a100/csr_layouts_rcm*.csv` |
| Tables of speedup, rate and counters | `sections/results.tex` | `analysis/make_results.py` | as above, and `quadro/` for the Quadro rows |
| FP16 controls | `sections/results.tex` | `analysis/make_results.py` | `a100/fp16_controls` |
| Bytes-per-load controls of CSR-r16 | `sections/results.tex` | `analysis/make_results.py` | `a100/r16_loads`, `quadro/r16_loads` |
| Gather without RCM | `sections/results.tex` | `analysis/make_results.py` | `a100/k_study_original`, `a100/csr_layouts.csv` |
| Fig. layouts | `figures/layouts.pdf` | `analysis/make_layout_figure.py --output-dir output/paper` | none |
| Tables of the matrices | `output/dataset.tex`, `output/dataset_matrices.tex` | `analysis/make_dataset_tex.py` | `matrices/manifest.csv`, `matrices/characterization*` |

## What is where

```
src/bench_csr_width.cu    the benchmark and every kernel
src/characterize_matrix.cpp  the structural pass over a matrix, without a GPU
src/device_report.cu      the constants of a GPU, written to device.txt
include/                  CSR reader, RCM, timers and value types
analysis/                 the scripts of the table above
experiments/              campaign runner, smoke test, matrix lists, Slurm jobs
matrices/                 download script, list of the 74 matrices, their characterization
data/results/             the measurements of the paper, a100/ and quadro/
data/matrices.sha256      the checksum of every matrix
data/smoke_matrices/      the two matrices of level 2
```

In `data/results`, `csr_layouts_rcm.csv` is the campaign after reordering, `_tpr` adds every layout at every
`T`, and `csr_layouts.csv` is the original ordering. `k_study*` holds the counters exported from Nsight
Compute, `k_study_rcm` after reordering and `k_study_original` in the original ordering. `fp16_controls` repeats FP16
with FP16 accumulation and with CSR-r16 unrolled by 16. `r16_loads` times CSR-r16 beside three builds that change
only how its offsets are loaded (`experiments/slurm/r16_loads.sbatch`, summarized by `analysis/r16_loads.py`).
`element_major` and `unbounded` are earlier implementations of CSR-lp.

## Setup of the published measurements

| | A100 | Quadro RTX 5000 |
|---|---|---|
| GPU | NVIDIA A100-SXM-64GB | NVIDIA Quadro RTX 5000 |
| CUDA, driver, cuSPARSE | see `data/results/a100/device.txt` | see `data/results/quadro/device.txt` |

The build uses `-O3` and the compute capability of the GPU, as set in `CMakeLists.txt`. It needs CMake 3.20
or later and the CUDA toolkit with cuSPARSE.

## Matrices

The matrices are from the SuiteSparse Matrix Collection (https://sparse.tamu.edu), T. A. Davis and Y. Hu,
ACM Transactions on Mathematical Software 38(1), 2011, under CC BY 4.0. A copy with their permutations is in
the data record, https://doi.org/10.5281/zenodo.24060361. This repository stores only `dawson5` and `rim`, for the quick check of level 2.

## License

MIT, see `LICENSE`. The license does not cover the matrices.
