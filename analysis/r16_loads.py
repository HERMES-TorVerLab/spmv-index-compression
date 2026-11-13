#!/usr/bin/env python3
"""K/K_ref of CSR-r16 and of its bytes-per-load controls, from experiments/slurm/r16_loads.sbatch.

Every build times CSR-i32 and its own CSR-r16 on the same matrix, one after the other. The ratio is
taken at the threads per row fastest for CSR-i32 in that build, as in the paper, with the bytes each
build reports, so CSR-r16 wide counts its four bytes per offset.

  analysis/r16_loads.py <directory holding one sub-directory per binary>
"""
import csv
import math
import statistics
import sys
from pathlib import Path

BUILDS = (("bench_csr_width", "r16"), ("bench_csr_width_r16lane", "lane"),
          ("bench_csr_width_r16pair", "pair"), ("bench_csr_width_r16wide", "wide"))


def per_matrix(directory):
    """(matrix, precision) to the byte ratio and K/K_ref of CSR-r16 against CSR-i32 at the T of CSR-i32."""
    samples = {}
    for row in csv.DictReader(open(directory / "csr_layouts_rcm_tpr.csv")):
        samples[(row["matrix"], row["prec"], row["layout"], int(row["tpr"]))] = row
    best_threads = {}
    for row in csv.DictReader(open(directory / "csr_layouts_rcm.csv")):
        if (row["status"] == "ok" and row["kernel"] == "custom" and row["column_mode"] == "absolute"
                and row["column_bytes"] == "4"):
            best_threads[(row["matrix"], row["prec"])] = int(row["best_tpr"])
    ratios = {}
    for (matrix, precision), threads in best_threads.items():
        reference = samples.get((matrix, precision, "i32", threads))
        candidate = samples.get((matrix, precision, "r16", threads))
        if reference is None or candidate is None:
            continue
        speedup = float(reference["t_ms"]) / float(candidate["t_ms"])
        byte_ratio = float(reference["bytes_total"]) / float(candidate["bytes_total"])
        ratios[(matrix, precision)] = {"speedup": speedup, "byte_ratio": byte_ratio, "rate": speedup / byte_ratio,
                                       "difference": float(candidate["max_abs_diff"]),
                                       "bytes": float(candidate["bytes_total"]), "time": float(candidate["t_ms"])}
    return ratios


def geometric_mean(values):
    return math.exp(sum(math.log(value) for value in values) / len(values))


def main():
    root = Path(sys.argv[1])
    results = {label: per_matrix(root / name) for name, label in BUILDS if (root / name / "csr_layouts_rcm.csv").exists()}
    common = set.intersection(*(set(values) for values in results.values()))
    for precision in sorted({key[1] for key in common}, reverse=True):
        keys = sorted(key for key in common if key[1] == precision)
        print(f"{precision}, {len(keys)} matrices, geometric means")
        for label, values in results.items():
            chosen = [values[key] for key in keys]
            print(f"  {label:5s} q_ref/q {geometric_mean([v['byte_ratio'] for v in chosen]):.3f}"
                  f"  K/K_ref {geometric_mean([v['rate'] for v in chosen]):.3f}"
                  f"  speedup {geometric_mean([v['speedup'] for v in chosen]):.3f}"
                  f"  K/K_ref over r16 {statistics.median(values[k]['rate'] / results['r16'][k]['rate'] for k in keys):.3f}"
                  f"  largest |dy| {max(v['difference'] for v in chosen):.1e}")
        for key in keys:
            print("    " + f"{key[0]:20s} " + " ".join(f"{label} {values[key]['rate']:.3f}" for label, values in results.items()))


if __name__ == "__main__":
    main()
