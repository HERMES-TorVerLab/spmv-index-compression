#!/usr/bin/env python3
"""make_i16_results.py — Results section: 16-bit column indices at FP64, and FP32 values, against CSR-i32 and cuSPARSE.

Inputs
  --csv (default results/quadro_rtx5000/csr_layouts.csv), written by experiments/run_i16.sh
  --device (default the device.txt next to the CSV), written by build/device_report: the memory
           roof, the compute ceilings and the machine named in the setup paragraph come from it
  matrices/manifest.csv, matrices/characterization/<name>.csv
Outputs, under --output-dir (default output/)
  results.tex                    the whole Results section: setup, 16-bit indices at FP64, FP32 and the
                                 comparison of the two levers (every number computed here)
  results_i16_table.tex          per-matrix FP64 table, \\input by results.tex as --table-input
  figures/i16_admissibility.pdf  how many matrices admit a 16-bit column index, by domain
  figures/i16_speedup.pdf        FP64 speedup of CSR-r16 and cuSPARSE against the byte model and the ceiling
  figures/fp32_speedup.pdf       speedup over FP64 CSR-i32 of FP64 CSR-r16, FP32 CSR-i32 and FP32 CSR-r16
  figures/lane_accuracy.pdf      measured against predicted speedup of the lane-packed layout, all matrices\n  figures/roofline.pdf           two panels: the machine with every compute ceiling, and the FP64 window matrix by matrix

Two 16-bit layouts are distinguished: CSR-i16 stores absolute 16-bit column indices and is exact
only when n <= 2^16 (13 matrices), CSR-r16 stores a 32-bit base per row plus 16-bit offsets and is
exact when every row span fits 16 bits (26 matrices). The results use CSR-r16; CSR-i16 is reported
where both apply, to price the per-row bases.
One sentence per line in the generated LaTeX; parentheses instead of em-dashes.
"""
import argparse
import csv
import glob
import math
import os
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402

from device_profile import load_device  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CUSPARSE = ("cusparse_csr_alg1", "cusparse_csr_alg2")
TIER_ORDER = {"S": 0, "M": 1, "L": 2, "XL": 3}
DOMAIN_LABEL = {
    "structural": "Structural", "cfd": "CFD", "pde": "2D/3D PDE", "electromagnetics": "Electromagn.",
    "circuit_power": "Circuit/power", "optimization": "Optimization", "chemistry": "Quantum chem.", "graph": "Graph",
}
DOMAIN_PROSE = {
    "structural": "structural mechanics", "cfd": "CFD", "pde": "2D/3D PDE", "electromagnetics": "electromagnetics",
    "circuit_power": "circuit and power networks", "optimization": "optimization", "chemistry": "quantum chemistry",
    "graph": "graph",
}
MIB = 1048576.0

# statistics.quantiles arrived in Python 3.8 and the login node of a cluster is often older, so the
# exclusive method of the standard library is reproduced here and used only when it is missing. The
# same cut points mean the same numbers on either interpreter.
def _quantiles(data, n=4):
    ordered = sorted(data)
    count = len(ordered)
    if count < 2:
        raise statistics.StatisticsError("quantiles needs at least two data points")
    width = count + 1
    result = []
    for index in range(1, n):
        position = index * width // n
        position = 1 if position < 1 else (count - 1 if position > count - 1 else position)
        delta = index * width - position * n
        result.append((ordered[position - 1] * (n - delta) + ordered[position] * delta) / n)
    return result


quantiles = getattr(statistics, "quantiles", _quantiles)

# Where the structural pass is looked for, newest first. They hold the same matrices and differ only
# in the columns the tool wrote at the time, so the first complete one wins.
CHARACTERIZATION_DIRS = ("characterization_v3", "characterization_v2", "characterization")
# Machine constants of the GPU the campaign ran on; loaded by main() from the device.txt written by
# build/device_report next to the CSV, so that changing machine does not mean editing this file.
DEVICE = None
# Which machine the paper reports. One variable moves every default, so a second GPU needs no flags:
#   SPMV_DEVICE=another-device analysis/make_i16_results.py --output-dir <paper>
DEVICE_DIR = os.environ.get("SPMV_DEVICE", "a100")
# Optional PNG previews of the figures: set RESULTS_PREVIEW_DIR to a directory.
PREVIEW = Path(os.environ["RESULTS_PREVIEW_DIR"]) if os.environ.get("RESULTS_PREVIEW_DIR") else None

# Reference categorical palette, slots 1-3 (validated all-pairs in light mode), plus neutral inks.
SERIES_BLUE = "#2a78d6"
SERIES_ORANGE = "#eb6834"
SERIES_AQUA = "#1baf7a"
NEUTRAL_FILL = "#c9c8c1"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3dd"

# Custom configurations compared against FP64 CSR-i32: (key, label, precision, layout label in the CSV).
PRECISION_CONFIGURATIONS = (
    ("fp64_i32", "FP64 CSR-i32", "fp64", "i32"),
    ("fp64_r16", "FP64 CSR-r16", "fp64", "r16"),
    ("fp64_lp", "FP64 CSR-lp", "fp64", "lp"),
    ("fp32_i32", "FP32 CSR-i32", "fp32", "i32"),
    ("fp32_r16", "FP32 CSR-r16", "fp32", "r16"),
    ("fp32_lp", "FP32 CSR-lp", "fp32", "lp"),
    ("fp16_i32", "FP16 CSR-i32", "fp16", "i32"),
    ("fp16_r16", "FP16 CSR-r16", "fp16", "r16"),
    ("fp16_lp", "FP16 CSR-lp", "fp16", "lp"),
)
# cuSPARSE has no FP16 SpMV on this release: cusparseSpMV returns "operation not supported".
CUSPARSE_PRECISIONS = (("cusparse_fp64", "FP64 cuSPARSE", "fp64"), ("cusparse_fp32", "FP32 cuSPARSE", "fp32"))
PRECISION_COLOUR = {"fp64": SERIES_BLUE, "fp32": SERIES_ORANGE, "fp16": SERIES_AQUA}
LAYOUT_MARKER = {"i32": "o", "r16": "s", "lp": "^"}

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "pdf.fonttype": 42, "font.family": "serif",
                     "axes.edgecolor": INK_SECONDARY, "axes.labelcolor": INK_PRIMARY,
                     "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY})


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def read_rows(path):
    with open(path) as handle:
        return list(csv.DictReader(handle))


def label_of(row):
    if row["kernel"] != "custom":
        return row["kernel"]
    if row["column_mode"] == "lane-packed":
        return "lp"
    return ("r" if row["column_mode"] == "row-base" else "i") + str(8 * int(row["column_bytes"]))


def tex_escape(text):
    return text.replace("_", r"\_")


def load_measurements(csv_path):
    """(matrix, precision) -> {label: row}, admissible configurations with the cumulative row pointer.

    The CSV is appended to, and a campaign that died in the middle of a matrix leaves its partial rows
    behind, because the resume skips a matrix only once its "end" line is in the log. The later rows
    win, which is what one wants, but a silent overwrite hides a run that was not what it looked like,
    so the duplicates are counted and reported."""
    table = {}
    duplicates = 0
    for row in read_rows(csv_path):
        if row["status"] != "ok" or row["row_encoding"] != "cumulative":
            continue
        entries = table.setdefault((row["matrix"], row["prec"]), {})
        label = label_of(row)
        duplicates += label in entries
        entries[label] = row
    if duplicates:
        print(f"note: {csv_path} has {duplicates} rows superseded by a later run of the same "
              "configuration, the last of each was used")
    return table


def load_characterization(directory=None):
    """The structural pass over every matrix, which depends on the sparsity pattern alone and is
    therefore machine independent.

    Without an explicit directory the first complete one of CHARACTERIZATION_DIRS is used. They
    differ only in how many columns src/characterize_matrix.cpp wrote when they were produced: the
    later passes add the work-side quantities (lane occupancy, tail ratio) and the gather traffic
    counted in sectors, which the applicability test and the extended model need. A pass that is
    still running is skipped rather than half read."""
    manifest = {row["name"]: row for row in read_rows(ROOT / "matrices" / "manifest.csv")}
    candidates = [Path(directory)] if directory else [ROOT / "matrices" / name for name in CHARACTERIZATION_DIRS]
    directory = next((path for path in candidates if len(glob.glob(str(path / "*.csv"))) >= len(manifest)),
                     candidates[-1])
    characterization = {}
    for path in glob.glob(str(directory / "*.csv")):
        row = read_rows(path)[0]
        row.update(domain=manifest[row["name"]]["category"], tier=manifest[row["name"]]["tier"])
        characterization[row["name"]] = row
    return characterization


def load_instruction_counts(path):
    """Instructions per nonzero of every kernel, read out of the SASS by extract_instruction_counts.py."""
    if not path.exists():
        return {}
    return {(row["layout"], row["precision"], int(row["threads_per_row"])): float(row["instructions_per_nnz"])
            for row in read_rows(path)}


def threads_for(characterization):
    """The threads per row the kernel would be given without measuring anything, the power of two
    nearest the mean row length inside the range the kernel supports."""
    length = max(float(characterization["row_length_mean"]), 2.0)
    return min(32, max(2, 2 ** round(math.log2(length))))


def is_applicable(characterization, resident_threads=None):
    """Whether the byte model has a premise on this matrix. One vector runs per row, so the machine
    holds resident_threads / T of them at once. When the longest row alone outlasts every other row
    spread over those slots, no layout saturates the memory system and a ratio of byte counts predicts
    nothing. Both sides of the test come from the pattern and the launch configuration."""
    threads = threads_for(characterization)
    # 49152 is what the reference Turing device reports (48 multiprocessors of 1024 threads); it is
    # only used when device.txt predates the threads_per_sm field.
    resident = resident_threads or (DEVICE.resident_threads() if DEVICE else 0.0) or 49152.0
    concurrent = resident / threads
    tail = float(characterization[f"tail_ratio_t{threads}"])
    return tail * concurrent / int(characterization["rows"]) <= 1.0


def best_cusparse(entries):
    return min((entry for name, entry in entries.items() if name in CUSPARSE), key=lambda entry: float(entry["t_ms"]))


# ---------------------------------------------------------------------------
# Per-matrix records
# ---------------------------------------------------------------------------

def summarize_fp64(table, characterization):
    records = []
    for (matrix, precision), entries in table.items():
        if precision != "fp64" or "i32" not in entries or "r16" not in entries:
            continue
        baseline = entries["i32"]
        narrow = entries["r16"]
        cusparse_ms = float(best_cusparse(entries)["t_ms"])
        records.append({
            "matrix": matrix,
            "domain": characterization[matrix]["domain"],
            "tier": characterization[matrix]["tier"],
            "nnz": int(baseline["nnz"]),
            "row_length": float(characterization[matrix]["row_length_mean"]),
            "imbalance": int(characterization[matrix]["row_length_max"]) / float(characterization[matrix]["row_length_mean"]),
            "memory_i32": float(baseline["bytes_total"]) / MIB,
            "memory_i16": float(narrow["bytes_total"]) / MIB,
            "gflops_i32": float(baseline["gflops"]),
            "gflops_i16": float(narrow["gflops"]),
            "gflops_cusparse": 2.0 * int(baseline["nnz"]) / (cusparse_ms * 1e-3) / 1e9,
            "speedup": float(baseline["t_ms"]) / float(narrow["t_ms"]),
            "predicted": float(baseline["bytes_total"]) / float(narrow["bytes_total"]),
            "over_cusparse": cusparse_ms / float(narrow["t_ms"]),
            "i32_over_cusparse": cusparse_ms / float(baseline["t_ms"]),
            "ceiling": float(characterization[matrix]["speedup_ceiling"]),
            "intensity_i32": 2.0 / float(baseline["bytes_per_nnz"]),
            "intensity_r16": 2.0 / float(narrow["bytes_per_nnz"]),
            "intensity_bound": 2.0 / (8.0 + float(baseline["bytes_per_nnz"]) - 12.0),
            "memory_absolute": float(entries["i16"]["bytes_total"]) / MIB if "i16" in entries else None,
            "gflops_absolute": float(entries["i16"]["gflops"]) if "i16" in entries else None,
            "memory_lane": float(entries["lp"]["bytes_total"]) / MIB if "lp" in entries else None,
            "gflops_lane": float(entries["lp"]["gflops"]) if "lp" in entries else None,
            "speedup_lane": float(baseline["t_ms"]) / float(entries["lp"]["t_ms"]) if "lp" in entries else None,
            # the packed width depends on the pattern, so CSR-lp has one prediction per matrix and no curve
            "predicted_lane": (float(baseline["bytes_total"]) / float(entries["lp"]["bytes_total"])
                               if "lp" in entries else None),
            "absolute_speedup": float(baseline["t_ms"]) / float(entries["i16"]["t_ms"]) if "i16" in entries else None,
            "absolute_predicted": (float(baseline["bytes_total"]) / float(entries["i16"]["bytes_total"])
                                   if "i16" in entries else None),
        })
    records.sort(key=lambda record: (TIER_ORDER[record["tier"]], record["nnz"]))
    return records


def summarize_lane(table, characterization):
    """Per matrix at FP64: the lane bit-packed layout against CSR-i32 and cuSPARSE. Unlike CSR-r16 it is
    exact on every matrix, so this covers the whole set."""
    records = []
    for (matrix, precision), entries in table.items():
        if precision != "fp64" or "i32" not in entries or "lp" not in entries:
            continue
        baseline, packed = entries["i32"], entries["lp"]
        cusparse_ms = float(best_cusparse(entries)["t_ms"]) if any(name in entries for name in CUSPARSE) else None
        row_length = float(characterization[matrix]["row_length_mean"])
        records.append({
            "matrix": matrix,
            "tier": characterization[matrix]["tier"],
            "domain": characterization[matrix]["domain"],
            "row_length": row_length,
            "imbalance": int(characterization[matrix]["row_length_max"]) / row_length,
            "bits": 8.0 * float(packed["bytes_columns"]) / float(packed["nnz"]),
            "speedup": float(baseline["t_ms"]) / float(packed["t_ms"]),
            "predicted": float(baseline["bytes_total"]) / float(packed["bytes_total"]),
            "over_cusparse": cusparse_ms / float(packed["t_ms"]) if cusparse_ms else None,
            "against_r16": (float(entries["r16"]["t_ms"]) / float(packed["t_ms"])) if "r16" in entries else None,
            "bytes": float(packed["bytes_per_nnz"]),
            "bytes_r16": float(entries["r16"]["bytes_per_nnz"]) if "r16" in entries else None,
            "bit_identical": float(packed["max_abs_diff"]) == 0.0,
        })
    records.sort(key=lambda record: record["row_length"])
    return records


def summarize_precisions(table, characterization):
    """Per matrix: speedup over FP64 CSR-i32 (measured and byte model) and bytes per nonzero of every configuration."""
    records = []
    for matrix in sorted({matrix for matrix, _ in table}):
        entries_of = {precision: table.get((matrix, precision), {}) for precision in ("fp64", "fp32", "fp16")}
        if any(layout not in entries_of[precision] for _, _, precision, layout in PRECISION_CONFIGURATIONS):
            continue
        reference = entries_of["fp64"]["i32"]
        record = {"matrix": matrix, "row_length": float(characterization[matrix]["row_length_mean"])}
        for key, _, precision, layout in PRECISION_CONFIGURATIONS:
            entry = entries_of[precision][layout]
            record[key] = float(reference["t_ms"]) / float(entry["t_ms"])
            record[key + "_predicted"] = float(reference["bytes_total"]) / float(entry["bytes_total"])
            record[key + "_bytes"] = float(entry["bytes_per_nnz"])
            record[key + "_throughput"] = float(entry["bw_gbs"])
            record[key + "_gflops"] = float(entry["gflops"])
            record[key + "_threads"] = int(entry["best_tpr"])
        record["applicable"] = is_applicable(characterization[matrix])
        for key, _, precision in CUSPARSE_PRECISIONS:
            fastest = best_cusparse(table[(matrix, precision)])
            record[key] = float(reference["t_ms"]) / float(fastest["t_ms"])
            record[key + "_bytes"] = float(table[(matrix, precision)]["i32"]["bytes_per_nnz"])
            record[key + "_gflops"] = float(fastest["gflops"])
        records.append(record)
    return records


def load_rcm_measurements(csv_path, original_csv=None):
    """Each matrix in its original ordering and after a symmetric RCM permutation, keyed by both.

    Two shapes of campaign exist. An exploratory run measures both orderings and writes them to one
    file with the ordering column telling them apart. A production run measures the permutation alone
    and leaves the original ordering to the main campaign, which already has it, so nothing is
    measured twice. Passing the main campaign as original_csv covers the second shape, and rows it
    would duplicate are dropped rather than overwriting what the reordered file already holds.

    The one-thread-per-row control shares the layout fields with CSR-i32, so only the vector kernel
    is kept.
    """
    table = {}
    rows = list(read_rows(csv_path))
    if original_csv is not None and not any(row.get("ordering") == "original" for row in rows):
        rows += [row for row in read_rows(original_csv) if row.get("ordering", "original") == "original"]
    for row in rows:
        if row["status"] != "ok":
            continue
        key = (row["matrix"], row["prec"], row["ordering"])
        entries = table.setdefault(key, {})
        if row["kernel"].startswith("cusparse"):
            previous = entries.get("cusparse")
            if previous is None or float(row["t_ms"]) < float(previous["t_ms"]):
                entries["cusparse"] = row
            continue
        if row["kernel"] != "custom":
            continue
        name = {"lane-packed": "lp", "row-base": "r16"}.get(row["column_mode"])
        if name is None and row["column_mode"] == "absolute":
            name = "i" + str(int(row["column_bytes"]) * 8)
        if name:
            entries[name] = row
    return table


def ordering_seconds(log_path):
    """Wall-clock cost of computing the permutations, as the campaign log recorded it."""
    if not log_path.exists():
        return None
    times = [float(value) for value in re.findall(r"ordering computed in ([0-9.]+) s",
                                                  log_path.read_text(errors="replace"))]
    return (statistics.median(times), max(times)) if times else None


def summarize_rcm(rcm_table, characterization, precision="fp64"):
    """Per matrix: whether each 16-bit layout is exact in either ordering, what the packed width
    becomes, and what reordering does to the kernels and to the byte model."""
    records = []
    for matrix in sorted({name for name, _, _ in rcm_table}):
        original = rcm_table.get((matrix, precision, "original"), {})
        reordered = rcm_table.get((matrix, precision, "rcm"), {})
        if "i32" not in original or "i32" not in reordered:
            continue
        rate = lambda entries, layout: float(entries["i32"]["t_ms"]) / float(entries[layout]["t_ms"])
        model = lambda entries, layout: float(entries["i32"]["bytes_total"]) / float(entries[layout]["bytes_total"])
        bits = lambda entries: 8.0 * float(entries["lp"]["bytes_columns"]) / float(entries["lp"]["nnz"])
        records.append({
            "matrix": matrix,
            "domain": characterization[matrix]["domain"],
            "tier": characterization[matrix]["tier"],
            "admissible_original": "r16" in original,
            "admissible_rcm": "r16" in reordered,
            "bits_original": bits(original),
            "bits_rcm": bits(reordered),
            "i32_gain": float(original["i32"]["t_ms"]) / float(reordered["i32"]["t_ms"]),
            "r16_speedup": rate(reordered, "r16") if "r16" in reordered else None,
            "r16_predicted": model(reordered, "r16") if "r16" in reordered else None,
            "r16_end_to_end": (float(original["i32"]["t_ms"]) / float(reordered["r16"]["t_ms"])
                               if "r16" in reordered else None),
            "lp_accuracy_original": rate(original, "lp") / model(original, "lp"),
            "lp_accuracy_rcm": rate(reordered, "lp") / model(reordered, "lp"),
            "r16_accuracy_rcm": (rate(reordered, "r16") / model(reordered, "r16")) if "r16" in reordered else None,
        })
    return records


def measurement_checks(table):
    """Bit identity of CSR-r16 and CSR-lp against CSR-i32 at every precision, and the worst relative
    deviation of cuSPARSE."""
    identical = {}
    cusparse_deviation = {}
    for (matrix, precision), entries in table.items():
        for layout in ("r16", "lp"):
            if layout in entries:
                identical.setdefault(precision, []).append(float(entries[layout]["max_abs_diff"]) == 0.0)
        for name in CUSPARSE:
            if name in entries:
                cusparse_deviation[precision] = max(cusparse_deviation.get(precision, 0.0),
                                                    float(entries[name]["max_rel_diff"]))
    return identical, cusparse_deviation


def warn_if_cusparse_fp16(csv_path):
    """The section states that cuSPARSE has no FP16 SpMV, which is true of the release measured on
    the Quadro RTX 5000 but is a property of the toolkit, not of the study. A newer cuSPARSE that
    does support it would be silently dropped from the comparison, so say so loudly."""
    supported = [row["matrix"] for row in read_rows(csv_path)
                 if row["prec"] == "fp16" and row["kernel"] in CUSPARSE and row["status"] == "ok"]
    if supported:
        print(f"WARNING: cuSPARSE ran FP16 SpMV on {len(supported)} matrices of {csv_path} "
              f"(first: {supported[0]}); the FP16 prose and CUSPARSE_PRECISIONS assume it cannot.")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def save_figure(figure, name, figures_dir):
    figure.savefig(figures_dir / f"{name}.pdf")
    if PREVIEW:
        figure.savefig(PREVIEW / f"{name}.png", dpi=220)


def style_axis(axis):
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def rcm_admissible(path):
    """Matrices on which CSR-r16 is exact after the symmetric permutation, from a reordered campaign.

    Whether every row of a matrix fits a 16-bit offset is a property of the matrix and of the
    permutation, not of the device, so this census can be taken from whichever campaign covers the
    whole set. The two machines agree on all 45 matrices they both reordered, which is the check that
    licenses reading it off one of them.

    A reordered campaign may hold both orderings in one file, so the ordering column is what decides,
    not the file name.
    """
    if not Path(path).exists():
        return None
    measured, exact = set(), set()
    for row in csv.DictReader(open(path, newline="")):
        if row.get("ordering") != "rcm" or row["kernel"] != "custom" or row["prec"] != "fp64":
            continue
        if row["row_encoding"] != "cumulative" or row["status"] not in ("ok", "not admissible"):
            continue
        measured.add(row["matrix"])
        if row["column_mode"] == "row-base" and row["status"] == "ok":
            exact.add(row["matrix"])
    return (measured, exact) if measured else None


def census_agreement(first, second):
    """Matrices reordered by both campaigns, and how many of them the two agree about.

    Exactness after the permutation depends on the matrix and the permutation, so the two machines
    have to produce the same answer. This is the check, and it is what licenses taking the census
    from whichever run covers the whole set.
    """
    left, right = rcm_admissible(first), rcm_admissible(second)
    if not left or not right:
        return None
    shared = left[0] & right[0]
    return len(shared), len(shared - ((left[1] ^ right[1]) & shared))


def admissibility_figure(characterization, figures_dir, reordered=None):
    """How many matrices admit a 16-bit column index, by domain, in each ordering.

    The three categories are nested rather than independent. A matrix with at most 2^16 columns takes
    an absolute 16-bit index and needs nothing else, one with more columns may still fit if the
    offsets are taken from a per-row base, and the rest fit in neither. The permutation cannot change
    the first category, since it does not change the number of columns, so the only boundary that
    moves is the second one, which is the comparison the figure is for.
    """
    present = [domain for domain in DOMAIN_LABEL
               if any(row["domain"] == domain for row in characterization.values())]
    orderings = [("original", None)]
    if reordered:
        orderings.append(("RCM", reordered[1]))

    def counts(domain, exact_after):
        rows = [row for row in characterization.values() if row["domain"] == domain]
        absolute = [row for row in rows if int(row["cols"]) <= 65536]
        wide = [row for row in rows if int(row["cols"]) > 65536]
        if exact_after is None:
            relative = [row for row in wide if float(row["rows_fitting_16bit"]) == 1.0]
        else:
            relative = [row for row in wide if row["name"] in exact_after]
        return len(absolute), len(relative), len(wide) - len(relative)

    # the two bars of a pair touch and the pairs are set apart, so the eye groups by domain rather
    # than by ordering
    height = 0.92 if reordered else 0.62
    pitch = len(orderings) + 0.8
    figure, axis = plt.subplots(figsize=(3.4, 0.95 + 0.17 * len(present) * len(orderings)))
    ticks, labels = [], []
    for index, domain in enumerate(present):
        for offset, (name, exact_after) in enumerate(orderings):
            position = -(index * pitch + offset)
            absolute, relative, neither = counts(domain, exact_after)
            axis.barh(position, absolute, height=height, color=SERIES_BLUE, edgecolor="white",
                      linewidth=1.0, label="absolute" if index == offset == 0 else None)
            axis.barh(position, relative, left=absolute, height=height, color=SERIES_AQUA,
                      edgecolor="white", linewidth=1.0, hatch="////",
                      label="per-row base only" if index == offset == 0 else None)
            axis.barh(position, neither, left=absolute + relative, height=height, color=NEUTRAL_FILL,
                      edgecolor="white", linewidth=1.0, label="neither" if index == offset == 0 else None)
            ticks.append(position)
            labels.append(DOMAIN_LABEL[domain] if offset == 0 else name)
    axis.set_ylim(-(len(present) - 1) * pitch - len(orderings) + 0.3, 0.7)
    axis.set_yticks(ticks, labels)
    for tick, label in zip(axis.get_yticklabels(), labels):
        if label in {name for name, _ in orderings[1:]}:
            tick.set_color(INK_SECONDARY)
            tick.set_fontsize(tick.get_fontsize() * 0.85)
    axis.set_xlabel("matrices")
    axis.xaxis.grid(True, color=GRID, linewidth=0.6)
    style_axis(axis)
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
                  handlelength=1.2, columnspacing=1.2)
    figure.tight_layout(pad=0.3, rect=(0.0, 0.0, 1.0, 1.0 - 0.30 / figure.get_figheight()))
    save_figure(figure, "i16_admissibility", figures_dir)
    plt.close(figure)
    totals = [counts(domain, None) for domain in present]
    admissible_by_domain = {domain: totals[index][0] + totals[index][1]
                            for index, domain in enumerate(present)}
    return (sum(t[0] for t in totals), sum(t[1] for t in totals), sum(t[2] for t in totals),
            admissible_by_domain)


def row_lengths_for_curves():
    return [10 ** (0.5 + index * 2.4 / 200) for index in range(201)]


def i16_speedup_figure(records, figures_dir):
    lengths = row_lengths_for_curves()
    figure, axis = plt.subplots(figsize=(3.4, 2.2))
    axis.plot(lengths, [(12 + 20 / length) / (8 + 20 / length) for length in lengths], color=INK_SECONDARY,
              linewidth=1.2, linestyle=(0, (4, 2)), label=r"ceiling $S_{\max}$")
    axis.plot(lengths, [(12 + 20 / length) / (10 + 24 / length) for length in lengths], color=SERIES_BLUE,
              linewidth=1.2, label=r"byte model $\hat S$")
    axis.axhline(1.0, color=INK_SECONDARY, linewidth=0.8)
    axis.scatter([record["row_length"] for record in records], [record["speedup"] for record in records], s=22,
                 marker="o", color=SERIES_BLUE, edgecolors="white", linewidths=0.8, zorder=3, label="CSR-r16")
    axis.scatter([record["row_length"] for record in records], [record["predicted_lane"] for record in records], s=22,
                 marker="^", facecolor="none", edgecolors=SERIES_AQUA, linewidths=0.9, zorder=2,
                 label=r"CSR-lp, $\hat S$")
    axis.scatter([record["row_length"] for record in records], [record["speedup_lane"] for record in records], s=22,
                 marker="^", color=SERIES_AQUA, edgecolors="white", linewidths=0.8, zorder=3, label="CSR-lp")
    axis.scatter([record["row_length"] for record in records], [record["i32_over_cusparse"] ** -1 for record in records],
                 s=22, marker="s", color=SERIES_ORANGE, edgecolors="white", linewidths=0.8, zorder=3,
                 label="cuSPARSE")
    axis.set_xscale("log")
    axis.set_xlabel(r"mean row length $\ell$")
    axis.set_ylabel("FP64 speedup over CSR-i32")
    axis.yaxis.grid(True, color=GRID, linewidth=0.6)
    style_axis(axis)
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False,
                  handlelength=1.6, columnspacing=0.9, handletextpad=0.4)
    axis.set_xlim(lengths[0], lengths[-1])
    axis.set_ylim(0.75, 1.55)
    figure.tight_layout(pad=0.3, rect=(0.0, 0.0, 1.0, 0.9))
    save_figure(figure, "i16_speedup", figures_dir)
    plt.close(figure)


def draw_machine_roof(axis, table, characterization, bandwidth, title):
    """The whole machine: the compute ceiling of every precision and the median position of
    CSR-i32, CSR-r16 and the index-free bound, which shows that no index encoding leaves the
    memory-bound region."""
    ceilings = tuple((precision, label, DEVICE.peak_gflops[precision]) for precision, label in
                     (("fp64", "FP64 peak"), ("fp32", "FP32 peak"), ("fp16", "FP16 peak")))
    colours = {"fp64": SERIES_BLUE, "fp32": SERIES_ORANGE, "fp16": SERIES_AQUA}
    wide = [10 ** (-1.1 + index * 3.6 / 400) for index in range(401)]
    axis.plot(wide, [bandwidth * intensity for intensity in wide], color=INK_SECONDARY, linewidth=1.3, zorder=2)
    axis.annotate(f"memory roof {bandwidth:.0f} GB/s", xy=(0.4, bandwidth * 0.28), color=INK_SECONDARY,
                      fontsize=6, rotation=29)
    for precision, label, peak in ceilings:
        axis.plot([peak / bandwidth, wide[-1]], [peak, peak], color=colours[precision], linewidth=1.1,
                      linestyle=(0, (4, 2)), zorder=1)
        label_y = peak * (0.6 if precision == "fp64" else 1.3)
        axis.annotate(f"{label} {peak:,.0f}", xy=(wide[-1] * 0.95, label_y), color=colours[precision],
                          fontsize=6, ha="right")
        baseline, narrow, bound = [], [], []
        for matrix in characterization:
            entries = table.get((matrix, precision), {})
            if "i32" not in entries or "r16" not in entries:
                continue
            q32 = float(entries["i32"]["bytes_per_nnz"])
            baseline.append((2.0 / q32, float(entries["i32"]["gflops"])))
            narrow.append((2.0 / float(entries["r16"]["bytes_per_nnz"]), float(entries["r16"]["gflops"])))
            bound.append(2.0 / (q32 - 4.0))
        if not baseline:
            continue
        median_point = lambda points, index: statistics.median(point[index] for point in points)
        axis.scatter([median_point(baseline, 0)], [median_point(baseline, 1)], s=26, marker="o",
                         color=colours[precision], edgecolors="white", linewidths=0.6, zorder=4,
                         label="CSR-i32" if precision == "fp64" else None)
        axis.scatter([median_point(narrow, 0)], [median_point(narrow, 1)], s=26, marker="o", facecolor="none",
                         edgecolors=colours[precision], linewidths=1.0, zorder=4,
                         label="CSR-r16" if precision == "fp64" else None)
        axis.scatter([statistics.median(bound)], [bandwidth * statistics.median(bound)], s=26, marker="x",
                         color=colours[precision], linewidths=0.9, zorder=5,
                         label="index-free bound" if precision == "fp64" else None)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(0.08, 400)
    axis.set_ylim(min(0.08 * bandwidth, 0.7 * min(DEVICE.peak_gflops.values())),
                  3.0 * max(DEVICE.peak_gflops.values()))
    axis.set_xlabel(r"arithmetic intensity $I$ [flop/byte]")
    axis.set_ylabel("performance [GFLOP/s]")
    axis.grid(True, which="major", color=GRID, linewidth=0.5)
    style_axis(axis)
    axis.legend(loc="lower right", frameon=False, fontsize=6, handlelength=1.2, handletextpad=0.3)
    axis.set_title(title, fontsize=7, loc="left", color=INK_PRIMARY)


def draw_fp64_window(axis, records, bandwidth, title):
    """The FP64 window, matrix by matrix, which shows how far the matrices move."""
    intensities = [value for record in records
                   for value in (record["intensity_i32"], record["intensity_r16"], record["intensity_bound"])]
    rates = [value for record in records for value in (record["gflops_i32"], record["gflops_i16"])]
    left = 0.02 * math.floor(min(intensities) / 0.02)
    right = 0.02 * math.ceil(max(intensities) / 0.02)
    narrow_window = [left + index * (right - left) / 200 for index in range(201)]
    axis.plot(narrow_window, [bandwidth * intensity for intensity in narrow_window], color=INK_SECONDARY,
              linewidth=1.3)
    axis.annotate(r"$P = B \cdot I$", xy=(left + 0.04 * (right - left), bandwidth * (left + 0.04 * (right - left))),
                  xytext=(4, 6), textcoords="offset points", color=INK_SECONDARY, fontsize=7, rotation=33)
    for record in records:
        axis.plot([record["intensity_i32"], record["intensity_r16"]], [record["gflops_i32"], record["gflops_i16"]],
                  color=GRID, linewidth=0.9, zorder=1)
    axis.scatter([record["intensity_bound"] for record in records],
                 [bandwidth * record["intensity_bound"] for record in records], s=26, marker="x",
                 color=SERIES_BLUE, linewidths=0.9, zorder=3, label="index-free bound")
    axis.scatter([record["intensity_i32"] for record in records], [record["gflops_i32"] for record in records],
                 s=22, marker="o", color=SERIES_BLUE, edgecolors="white", linewidths=0.8, zorder=4, label="CSR-i32")
    axis.scatter([record["intensity_r16"] for record in records], [record["gflops_i16"] for record in records],
                 s=22, marker="o", facecolor="none", edgecolors=SERIES_BLUE, linewidths=1.0, zorder=4,
                 label="CSR-r16")
    axis.set_xlim(left, right)
    axis.set_ylim(0.8 * min(rates), max(1.1 * max(rates), bandwidth * right))
    axis.set_xlabel(r"arithmetic intensity $I$ [flop/byte]")
    axis.set_ylabel("FP64 performance [GFLOP/s]")
    axis.grid(True, which="major", color=GRID, linewidth=0.5)
    style_axis(axis)
    axis.legend(loc="upper left", frameon=False, fontsize=6, handlelength=1.2, handletextpad=0.3)
    axis.set_title(title, fontsize=7, loc="left", color=INK_PRIMARY)


def roofline_panels(records, table, characterization, figures_dir, bandwidth=None):
    """Two figures from the same panels: both panels side by side for the archive build, and
    the machine panel alone, one column wide, for the conference build."""
    bandwidth = bandwidth or DEVICE.read_bw_gbs
    figure, (overview, zoom) = plt.subplots(1, 2, figsize=(7.0, 2.2))
    draw_machine_roof(overview, table, characterization, bandwidth, "A. the machine (medians per precision)")
    draw_fp64_window(zoom, records, bandwidth, "B. FP64, one pair of points per matrix")
    figure.tight_layout(pad=0.4, w_pad=1.6, rect=(0.0, 0.0, 0.985, 1.0))
    save_figure(figure, "roofline", figures_dir)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(3.4, 2.3))
    draw_machine_roof(axis, table, characterization, bandwidth, "medians per precision")
    figure.tight_layout(pad=0.3)
    save_figure(figure, "roofline_machine", figures_dir)
    plt.close(figure)


def lane_accuracy_figure(records, figures_dir, imbalance_threshold=10.0):
    """Measured against predicted speedup of CSR-lp over CSR-i32, one point per matrix.
    Matrices whose longest row is far above the mean fall below the identity line."""
    figure, axis = plt.subplots(figsize=(3.4, 2.5))
    limits = (0.4, 1.35)
    axis.plot(limits, limits, color=INK_SECONDARY, linewidth=1.2)
    axis.annotate(r"$S = \hat S$", xy=(1.22, 1.26), color=INK_SECONDARY, fontsize=7, rotation=45)
    balanced = [record for record in records if record["imbalance"] <= imbalance_threshold]
    skewed = [record for record in records if record["imbalance"] > imbalance_threshold]
    axis.scatter([record["predicted"] for record in balanced], [record["speedup"] for record in balanced], s=22,
                 marker="o", color=SERIES_BLUE, edgecolors="white", linewidths=0.8, zorder=3,
                 label=rf"$\ell_{{\max}}/\ell \leq {imbalance_threshold:.0f}$")
    axis.scatter([record["predicted"] for record in skewed], [record["speedup"] for record in skewed], s=22,
                 marker="^", color=SERIES_ORANGE, edgecolors="white", linewidths=0.8, zorder=3,
                 label=rf"$\ell_{{\max}}/\ell > {imbalance_threshold:.0f}$")
    axis.set_xlim(*limits)
    axis.set_ylim(*limits)
    axis.set_xlabel(r"byte-model prediction $\hat S$")
    axis.set_ylabel(r"measured speedup $S$")
    axis.grid(True, color=GRID, linewidth=0.6)
    style_axis(axis)
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
                  handlelength=1.4, columnspacing=1.0, handletextpad=0.3)
    figure.tight_layout(pad=0.3, rect=(0.0, 0.0, 1.0, 0.9))
    save_figure(figure, "lane_accuracy", figures_dir)
    plt.close(figure)


def performance_figure(records, figures_dir):
    """Absolute performance of every layout at every value precision, median over the matrices, each bar
    against the memory roof $2B/q$ that its own traffic allows."""
    layouts = (("i32", "CSR-i32"), ("r16", "CSR-r16"), ("lp", "CSR-lp"), ("cusparse", "cuSPARSE"))
    precisions = ("fp64", "fp32", "fp16")
    width = 0.26
    figure, axis = plt.subplots(figsize=(3.4, 2.0))
    for offset, precision in enumerate(precisions):
        centres, heights, lower, upper, roofs = [], [], [], [], []
        for position, (layout, _) in enumerate(layouts):
            key = f"{precision}_{layout}" if layout != "cusparse" else f"cusparse_{precision}"
            if key + "_gflops" not in records[0]:
                continue
            rates = sorted(record[key + "_gflops"] for record in records)
            quartiles = quantiles(rates, n=4)
            centres.append(position + (offset - 1) * width)
            heights.append(statistics.median(rates))
            lower.append(statistics.median(rates) - quartiles[0])
            upper.append(quartiles[2] - statistics.median(rates))
            roofs.append(2.0 * DEVICE.read_bw_gbs / statistics.median(record[key + "_bytes"]
                                                                     for record in records))
        axis.bar(centres, heights, width, color=PRECISION_COLOUR[precision], label=precision.upper(), zorder=2)
        axis.errorbar(centres, heights, yerr=[lower, upper], fmt="none", ecolor=INK_SECONDARY, elinewidth=0.8,
                      capsize=1.6, zorder=3)
        for centre, roof in zip(centres, roofs):
            axis.plot([centre - width / 2, centre + width / 2], [roof, roof], color=INK_PRIMARY, linewidth=1.0,
                      zorder=4)
    axis.plot([], [], color=INK_PRIMARY, linewidth=1.0, label=r"memory roof $2B/q$")
    axis.set_xticks(range(len(layouts)))
    axis.set_xticklabels([label for _, label in layouts])
    axis.set_ylabel("performance [GFLOP/s]")
    axis.yaxis.grid(True, color=GRID, linewidth=0.6)
    style_axis(axis)
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False,
                  fontsize=6, handlelength=1.4, columnspacing=0.9, handletextpad=0.4)
    figure.tight_layout(pad=0.3, rect=(0.0, 0.0, 1.0, 0.9))
    save_figure(figure, "performance", figures_dir)
    plt.close(figure)


def precision_speedup_figure(records, figures_dir, instructions):
    """Measured against predicted speedup, one point per matrix and configuration, on the left with the
    byte count alone and on the right with the decoding cost of the compiled kernel added. Matrices the
    applicability test of Eq.~(applicable) excludes are drawn hollow, since no byte count speaks for them."""
    issue_rate = DEVICE.issue_rate()
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    limits = (0.85, 4.0)
    for axis, with_work in zip(axes, (False, True)):
        axis.plot(limits, limits, color=INK_SECONDARY, linewidth=1.2, zorder=2)
        for key, _, precision, layout in PRECISION_CONFIGURATIONS:
            if key == "fp64_i32":
                continue
            for admitted in (True, False):
                chosen = [record for record in records if record["applicable"] == admitted]
                if not chosen:
                    continue
                predicted = []
                for record in chosen:
                    reference = record["fp64_i32_bytes"] / DEVICE.read_bw_gbs
                    candidate = record[key + "_bytes"] / DEVICE.read_bw_gbs
                    if with_work and issue_rate:
                        reference += instructions.get(("i32", "fp64", record["fp64_i32_threads"]), 0.0) / issue_rate * 1e9
                        candidate += instructions.get((layout, precision, record[key + "_threads"]), 0.0) / issue_rate * 1e9
                    predicted.append(reference / candidate)
                measured = [record[key] for record in chosen]
                if admitted:
                    axis.scatter(predicted, measured, s=18, marker=LAYOUT_MARKER[layout],
                                 color=PRECISION_COLOUR[precision], edgecolors="white", linewidths=0.5, zorder=3)
                else:
                    axis.scatter(predicted, measured, s=18, marker=LAYOUT_MARKER[layout], facecolor="none",
                                 edgecolors=NEUTRAL_FILL, linewidths=0.8, zorder=1)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xticks([1.0, 1.5, 2.0, 3.0, 4.0])
        axis.set_yticks([1.0, 1.5, 2.0, 3.0, 4.0])
        axis.xaxis.set_major_formatter(ScalarFormatter())
        axis.yaxis.set_major_formatter(ScalarFormatter())
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.set_xlim(*limits)
        axis.set_ylim(*limits)
        axis.set_xlabel("predicted speedup over FP64 CSR-i32")
        axis.grid(True, color=GRID, linewidth=0.6)
        style_axis(axis)
        axis.set_title("B. with the decoding cost" if with_work else "A. bytes alone",
                       fontsize=7, loc="left", color=INK_PRIMARY)
    axes[0].set_ylabel("measured speedup")
    keys = [Line2D([], [], color=PRECISION_COLOUR[precision], marker="o", linestyle="none", markersize=4,
                   label=precision.upper()) for precision in ("fp64", "fp32", "fp16")]
    keys += [Line2D([], [], color=INK_SECONDARY, marker=LAYOUT_MARKER[layout], linestyle="none", markersize=4,
                    label="CSR-" + layout) for layout in ("i32", "r16", "lp")]
    keys += [Line2D([], [], color=NEUTRAL_FILL, marker="o", linestyle="none", markersize=4, markerfacecolor="none",
                    label="outside the test")]
    figure.legend(handles=keys, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=7, frameon=False,
                  fontsize=6, handlelength=1.2, columnspacing=0.9, handletextpad=0.3)
    figure.tight_layout(pad=0.4, w_pad=1.4, rect=(0.0, 0.0, 1.0, 0.89))
    save_figure(figure, "precision_speedup", figures_dir)
    plt.close(figure)


def span(values, digits=3):
    return f"${min(values):.{digits}f}$--${max(values):.{digits}f}$"


def scientific(value):
    exponent = int(math.floor(math.log10(value)))
    return rf"{value / 10 ** exponent:.1f}\cdot10^{{{exponent}}}"


def i16_table_tex(records):
    lines = ["% Generated by analysis/make_i16_results.py; do not edit by hand.",
             r"\begin{table*}[t]", r"\centering",
             r"\caption{CSR-i32, CSR-r16 and cuSPARSE at FP64 on the " + str(len(records)) +
             r" matrices whose row spans all fit 16 bits, in their original ordering."
             r" Memory is the device working set (values, column indices, row bases, row pointer, $x$ and $y$)."
             r" cuSPARSE is the faster of CSR\_ALG1 and CSR\_ALG2."
             r" A dash marks the matrices where absolute 16-bit indices are not exact ($n > 2^{16}$)."
             r" $S_{\mathrm{r16}}$ and $S_{\mathrm{lp}}$ are the measured speedups over CSR-i32, $\hat S$ the byte-model"
             r" prediction for CSR-r16, and $S_{\max}$ the ceiling of Eq.~\eqref{eq:smax}.}",
             r"\label{tab:results-i16}", r"\scriptsize", r"\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabular}{llrrrrrrrrrrrrrr}", r"\toprule",
             r" & & & & \multicolumn{4}{c}{memory [MiB]} & \multicolumn{5}{c}{GFLOP/s} & & & \\",
             r"\cmidrule(lr){5-8}\cmidrule(lr){9-13}",
             r"matrix & tier & nnz [M] & $\ell$ & i32 & i16 & r16 & lp & i32 & i16 & r16 & lp & cuSP & $S_{\mathrm{r16}}$ & $S_{\mathrm{lp}}$ & $S_{\max}$ \\",
             r"\midrule"]
    previous_tier = None
    for record in records:
        if previous_tier is not None and record["tier"] != previous_tier:
            lines.append(r"\addlinespace[2pt]")
        previous_tier = record["tier"]
        absolute_memory = (f"{record['memory_absolute']:.0f}" if record["memory_absolute"] is not None else "--")
        absolute_rate = (f"{record['gflops_absolute']:.1f}" if record["gflops_absolute"] is not None else "--")
        lines.append(
            rf"\texttt{{{tex_escape(record['matrix'])}}} & {record['tier']} & {record['nnz'] / 1e6:.1f} & {record['row_length']:.1f}"
            rf" & {record['memory_i32']:.0f} & {absolute_memory} & {record['memory_i16']:.0f} & {record['memory_lane']:.0f}"
            rf" & {record['gflops_i32']:.1f} & {absolute_rate} & {record['gflops_i16']:.1f} & {record['gflops_lane']:.1f}"
            rf" & {record['gflops_cusparse']:.1f}"
            rf" & {record['speedup']:.3f} & {record['speedup_lane']:.3f} & {record['ceiling']:.3f} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def precision_table_lines(records):
    """One block per value precision, then the vendor reference. The last column is the attained
    throughput relative to the memory bandwidth, which is what the prediction assumes unchanged."""
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Speedup over FP64 CSR-i32 on the " + str(len(records)) +
             r" matrices whose row spans all fit 16 bits, showing medians of the bytes per nonzero $q$, of the byte-model"
             r" prediction $\hat S$, of the measured speedup $S$ (with its range), of $S/\hat S$ and of the attained"
             r" throughput $R$ of Eq.~\eqref{eq:throughput} relative to $B$."
             r" FP16 stores the values and the vectors in half precision and accumulates in FP32."
             r" cuSPARSE uses the arrays of CSR-i32 of the same precision, and since its kernel differs from ours no"
             r" prediction is given. It has no FP16 mode.}",
             r"\label{tab:results-fp32}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lrrrcrr}", r"\toprule",
             r"layout & $q$ [B] & $\hat S$ & $S$ & range of $S$ & $S/\hat S$ & $R/B$ \\", r"\midrule"]
    previous_precision = None
    for key, label, precision, layout in PRECISION_CONFIGURATIONS:
        measured = [record[key] for record in records]
        predicted = [record[key + "_predicted"] for record in records]
        ratios = [value / prediction for value, prediction in zip(measured, predicted)]
        bytes_median = statistics.median(record[key + "_bytes"] for record in records)
        throughput = statistics.median(record[key + "_throughput"] for record in records) / DEVICE.read_bw_gbs
        if precision != previous_precision:
            if previous_precision is not None:
                lines.append(r"\addlinespace[2pt]")
            lines.append(rf"\multicolumn{{7}}{{l}}{{\emph{{{precision.upper()} values}}}} \\")
            previous_precision = precision
        name = label.split()[1]
        if key == "fp64_i32":
            lines.append(rf"{name} (reference) & {bytes_median:.2f} & $1$ & $1$ & -- & -- & {throughput:.2f} \\")
            continue
        lines.append(rf"{name} & {bytes_median:.2f} & {statistics.median(predicted):.3f} & {statistics.median(measured):.3f}"
                     rf" & {min(measured):.2f}--{max(measured):.2f} & {statistics.median(ratios):.3f}"
                     rf" & {throughput:.2f} \\")
    lines.append(r"\addlinespace[2pt]")
    for key, label, precision in CUSPARSE_PRECISIONS:
        measured = [record[key] for record in records]
        lines.append(rf"{label} & {statistics.median(record[key + '_bytes'] for record in records):.2f} & --"
                     rf" & {statistics.median(measured):.3f} & {min(measured):.2f}--{max(measured):.2f} & -- & -- \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return lines


def intro_tex():
    return r"""% Generated by analysis/make_i16_results.py; do not edit by hand.
\section{Results}
\label{sec:results}

This section measures how much narrowing the column index speeds up SpMV at unchanged values, how well the byte model predicts that gain, and how the lossless gain compares with the lossy gain of narrowing the values themselves.
"""


def setup_tex(record_count, absolute_count, identical, cusparse_deviation, timing):
    reps, warmup, min_sample_ms = timing
    all_identical = all(all(flags) for flags in identical.values())
    identity_text = ("which they are on every matrix at all three precisions" if all_identical
                     else "which is checked on every matrix")
    sample_text = (rf"each sample running enough consecutive products to last at least ${min_sample_ms:g}$~ms "
                   r"so that launch costs do not bias the sub-millisecond kernels of tier S"
                   if min_sample_ms > 0 else "each sample timing a single product")
    return rf"""
\subsection{{Setup}}
\label{{sec:results-setup}}

All measurements run on {DEVICE.description()}{DEVICE.toolkit()}.
We measure the four layouts of Sec.~\ref{{sec:model}} (CSR-i32, CSR-i16, CSR-r16 and CSR-lp), with values in FP64 ($v = 8$), FP32 ($v = 4$) or FP16 ($v = 2$) and the vectors $x$ and $y$ in the same precision as the values.
All of them run in the same GPU kernel, one vector per row with the number of threads per row swept over $\{{2,4,8,16,32\}}$ and the fastest kept, which is what makes the attained throughputs comparable in Eq.~\eqref{{eq:ratio}}.
At a given precision they keep the value array and the order of the nonzeros, so their results must be bit-identical, {identity_text}.

The vendor reference is \texttt{{cusparseSpMV}} with the CSR\_ALG1 and CSR\_ALG2 algorithms on the same device arrays as CSR-i32, and we always report the faster of the two.
\briefonly{{It sums in a different order and agrees with us to ${scientific(cusparse_deviation['fp64'])}$ in FP64 and ${scientific(cusparse_deviation['fp32'])}$ in FP32, relative to $\max_r |y_r|$.}}
\extonly{{cuSPARSE sums in a different order, and its result differs from ours by at most $\max_r |\Delta y_r| / \max_r |y_r| = {scientific(cusparse_deviation['fp64'])}$ in FP64 and ${scientific(cusparse_deviation['fp32'])}$ in FP32.}}
Each time is the median of ${reps}$ samples after ${warmup}$ warm-up products, {sample_text}.
The measurements cover the {record_count} matrices of Sec.~\ref{{sec:dataset}}, from all four tiers, on which CSR-r16 is exact, and CSR-i16 is exact on {absolute_count} of them.
"""


def i16_tex(records, counts, total_matrices, table_input, census=None):
    absolute_count, relative_only_count, _, admissible_by_domain = counts
    admissible = absolute_count + relative_only_count
    # the figure gains a second bar per domain only when a reordered campaign covers the whole set
    census_caption = " and ordering" if census else ""
    census_detail = ""
    if census:
        census_detail = (
            r" The permutation cannot move the first category, since it leaves $n$ unchanged, and it takes"
            rf" the total from {census['before']} to {census['after']} of the {total_matrices} matrices."
            rf" It is not monotone, widening {census['lost']} matrix past the bound while narrowing {census['gained']}."
            rf" Exactness belongs to the matrix and the permutation rather than to the device, and the two"
            rf" machines agree on all {census['agreed']} matrices both of them reordered.")
    speedups = [record["speedup"] for record in records]
    predicted = [record["predicted"] for record in records]
    accuracy = [record["speedup"] / record["predicted"] for record in records]
    over_cusparse = [record["over_cusparse"] for record in records]
    i32_over_cusparse = [record["i32_over_cusparse"] for record in records]
    savings = [100.0 * (1.0 - record["memory_i16"] / record["memory_i32"]) for record in records]
    ceilings = [record["ceiling"] for record in records]
    captures = [(record["speedup"] - 1.0) / (record["ceiling"] - 1.0) for record in records]
    beats_cusparse = sum(1 for value in over_cusparse if value > 1.0)
    within_five_percent = sum(1 for value in accuracy if abs(value - 1.0) <= 0.05)
    failures = len(records) - within_five_percent
    faster_than_i32 = sum(1 for value in speedups if value > 1.0)
    faster_text = ("on every matrix" if faster_than_i32 == len(records)
                   else f"on {faster_than_i32} of the {len(records)} matrices")
    absolute = [record for record in records if record["absolute_speedup"] is not None]
    absolute_count = len(absolute)
    absolute_gain = statistics.median(record["absolute_speedup"] / record["speedup"] for record in absolute)
    absolute_predicted = statistics.median(record["absolute_predicted"] / record["predicted"] for record in absolute)
    below_threshold = sorted((record for record in records if record["speedup"] / record["predicted"] < 0.95),
                             key=lambda record: record["speedup"] / record["predicted"])
    worst = max(records, key=lambda record: record["imbalance"])
    named = ", ".join(rf"\texttt{{{tex_escape(record['matrix'])}}} at ${record['speedup'] / record['predicted']:.3f}$"
                      for record in below_threshold[:3])
    if not below_threshold:
        failure_text = (f"The prediction is within $5\\,\\%$ of the measurement on every matrix, "
                        f"the worst case being ${min(accuracy):.3f}$.")
    else:
        failure_text = (
            f"The prediction misses the measurement by more than $5\\,\\%$ on {len(below_threshold)} of the "
            f"{len(records)} matrices ({named}).\n"
            f"{'That matrix is' if len(below_threshold) == 1 else 'Those matrices include'} the most imbalanced of the set, "
            f"whose longest row holds ${worst['imbalance']:.0f}$ times the mean, so part of the time does not scale with the bytes.")
    largest_count = max(admissible_by_domain.values())
    leading = [(domain, count) for domain, count in sorted(admissible_by_domain.items(), key=lambda item: -item[1])
               if count >= 0.5 * largest_count]
    leading_names = [f"{DOMAIN_PROSE[domain]} ({count})" for domain, count in leading]
    leading_text = leading_names[0] if len(leading_names) == 1 else ", ".join(leading_names[:-1]) + " and " + leading_names[-1]
    excluded = [DOMAIN_PROSE[domain] for domain, count in admissible_by_domain.items() if count == 0]
    excluded_text = (f", while no {' or '.join(excluded)} matrix qualifies in its original ordering" if excluded else "")
    return rf"""
\subsection{{16-bit column indices at FP64}}
\label{{sec:results-i16}}

We first compare CSR-r16 with CSR-i32 at unchanged FP64 values.
\extonly{{Fig.~\ref{{fig:i16-admissibility}} shows on how many matrices the narrow index can be used at all.}}
Absolute 16-bit indices are exact on {absolute_count} of the {total_matrices} matrices, and the per-row base of Eq.~\eqref{{eq:r16-column}} extends this to {admissible}, mostly in {leading_text}{excluded_text}.

\begin{{extended}}
\begin{{figure}}[t]
  \centering
  \includegraphics[width=\columnwidth]{{figures/i16_admissibility.pdf}}
  \caption{{Matrices whose column indices are exact in 16 bits, by domain{census_caption}, as absolute indices ($n \le 2^{{16}}$), only as offsets from a per-row base ($s_r < 2^{{16}}$ in every row), or in neither form.{census_detail}}}
  \label{{fig:i16-admissibility}}
\end{{figure}}
\end{{extended}}

On those {len(records)} matrices, \extonly{{Table~\ref{{tab:results-i16}} and }}Fig.~\ref{{fig:i16-speedup}} report the effect.
CSR-r16 reduces the working set by {span(savings, 0)}\,\% and is faster than CSR-i32 {faster_text}, with a median speedup of ${statistics.median(speedups):.3f}$ (range {span(speedups)}).
The byte model of Eq.~\eqref{{eq:q-layouts}} predicts a median of ${statistics.median(predicted):.3f}$, and the ratio of measured to predicted speedup has median ${statistics.median(accuracy):.3f}$ (range {span(accuracy)}), within $5\,\%$ on {within_five_percent} of the {len(records)} matrices, so the gain of the narrow index is known from the matrix dimensions before any code is run.
Our CSR-i32 kernel is on par with the vendor library, which takes a median ${statistics.median(i32_over_cusparse):.3f}$ times its time, and CSR-r16 is faster than the better cuSPARSE algorithm on {beats_cusparse} of the {len(records)} matrices, by a median of ${statistics.median(over_cusparse):.3f}\times$.
Against the ceiling $S_{{\max}}$ of Eq.~\eqref{{eq:smax}}, whose median is ${statistics.median(ceilings):.2f}$ on these matrices, the capture $\kappa$ of Eq.~\eqref{{eq:capture}} has median ${statistics.median(captures):.2f}$, close to the $0.4$ that Sec.~\ref{{sec:model}} derives for a layout that saves two of the four index bytes.
\extonly{{In the plane of Fig.~\ref{{fig:roofline}}B these gains are the horizontal moves, and the crosses are where $\kappa = 1$ would put each matrix.}}

{failure_text}

\begin{{extended}}
On the {absolute_count} matrices with at most $2^{{16}}$ columns, where CSR-i16 is exact as well, it is faster than CSR-r16 by a median of ${absolute_gain:.3f}\times$ against a predicted ${absolute_predicted:.3f}\times$, which is exactly the $4m$ bytes of the bases, so the model prices the two 16-bit layouts apart as well as it prices them against CSR-i32.
\end{{extended}}

\begin{{figure}}[t]
  \centering
  \includegraphics[width=\columnwidth]{{figures/i16_speedup.pdf}}
  \caption{{FP64 speedup over CSR-i32 against the mean row length, on the {len(records)} matrices whose row spans all fit 16 bits.
  Circles are CSR-r16, filled triangles CSR-lp and squares the faster of cuSPARSE CSR\_ALG1 and CSR\_ALG2.
  The solid line is the byte-model prediction for CSR-r16 and the dashed line the ceiling $S_{{\max}}$ of any index compression, both closed forms in $\ell$ for a square matrix.
  CSR-lp has no such curve, since its packed width follows the sparsity pattern rather than the dimensions, so its prediction is one open triangle per matrix.}}
  \label{{fig:i16-speedup}}
\end{{figure}}

\begin{{extended}}
\input{{{table_input}}}
\end{{extended}}
"""


def fp32_tex(records):
    table_block = "\n".join(precision_table_lines(records))
    median_of = lambda key: statistics.median(record[key] for record in records)
    predicted_of = lambda key: statistics.median(record[key + "_predicted"] for record in records)
    ratio_of = lambda key: statistics.median(record[key] / record[key + "_predicted"] for record in records)
    throughput_of = lambda key: (statistics.median(record[key + "_throughput"] for record in records)
                                 / DEVICE.read_bw_gbs)
    index_gain = lambda precision: statistics.median(record[precision + "_r16"] / record[precision + "_i32"]
                                                     for record in records)
    over_cusparse_fp32 = [record["fp32_r16"] / record["cusparse_fp32"] for record in records]
    beats_cusparse_fp32 = sum(1 for value in over_cusparse_fp32 if value > 1.0)
    # Direct comparison of the two levers: FP32 CSR-i32 against FP64 CSR-r16.
    lever_measured = [record["fp32_i32"] / record["fp64_r16"] for record in records]
    lever_predicted = [record["fp32_i32_predicted"] / record["fp64_r16_predicted"] for record in records]
    lever_ratio = [measured / predicted for measured, predicted in zip(lever_measured, lever_predicted)]
    lever_faster = sum(1 for value in lever_measured if value > 1.0)
    lever_share = [(record["fp64_r16"] - 1.0) / (record["fp32_i32"] - 1.0) for record in records]
    half_lever = [record["fp16_i32"] / record["fp64_r16"] for record in records]
    listed = lambda values: ", ".join(values[:-1]) + " and " + values[-1]
    rates_of = lambda precision: listed([
        f"{statistics.median(record[f'{precision}_{layout}_gflops'] for record in records):.0f}"
        for layout in ("i32", "r16", "lp")])
    roofs_of = lambda precision: listed([
        f"{2.0 * DEVICE.read_bw_gbs / statistics.median(record[f'{precision}_{layout}_bytes'] for record in records):.0f}"
        for layout in ("i32", "r16", "lp")])
    faster_text = ("on every matrix" if lever_faster == len(records)
                   else f"on {lever_faster} of the {len(records)} matrices")
    return rf"""
\subsection{{Index width against value precision}}
\label{{sec:results-fp32}}

Halving the value precision is the usual way to reduce the bytes that SpMV moves, but unlike a narrower index it changes the computed product.
To quantify what this lossy lever buys compared with the lossless one, we repeat the three layouts with FP32 and FP16 values on the same {len(records)} matrices.
FP16 here stores the values and the vectors in half precision and takes the products and the sums in FP32, the storage and computation split of the accessor of Grützmacher et al.~\cite{{grutzmacher2021accessor}}, so at every precision the three layouts still differ only in the bytes they read and remain bit-identical to each other.
The vendor reference is cuSPARSE at FP64 and FP32, since its SpMV has no FP16 mode on this release.
All speedups in Table~\ref{{tab:results-fp32}}\extonly{{ and Fig.~\ref{{fig:precision-speedup}}}} are taken against FP64 CSR-i32.

{table_block}

In isolation, narrowing the values is the stronger lever, since CSR-i32 gains ${median_of('fp32_i32'):.3f}\times$ at FP32 and ${median_of('fp16_i32'):.3f}\times$ at FP16 (predictions ${predicted_of('fp32_i32'):.3f}\times$ and ${predicted_of('fp16_i32'):.3f}\times$), against ${median_of('fp64_r16'):.3f}\times$ for a 16-bit index at FP64.
The two levers compound, and applying both gives ${median_of('fp32_r16'):.3f}\times$ at FP32 and ${median_of('fp16_r16'):.3f}\times$ at FP16 with CSR-r16, and ${median_of('fp32_lp'):.3f}\times$ and ${median_of('fp16_lp'):.3f}\times$ with CSR-lp, which needs no admissibility condition.
Narrowing the index is worth more as the values shrink, ${index_gain('fp64'):.3f}\times$ at FP64, ${index_gain('fp32'):.3f}\times$ at FP32 and ${index_gain('fp16'):.3f}\times$ at FP16, because the four index bytes are a larger share of the traffic when a value takes four or two bytes instead of eight.
cuSPARSE gains ${median_of('cusparse_fp32') / median_of('cusparse_fp64'):.3f}\times$ from FP64 to FP32, close to the ${median_of('fp32_i32'):.3f}\times$ of CSR-i32, and FP32 CSR-r16 is faster than FP32 cuSPARSE on {beats_cusparse_fp32} of the {len(records)} matrices (median ${statistics.median(over_cusparse_fp32):.3f}\times$).

The same table shows where the byte model stops holding, and why.
The ratio of measured to predicted speedup stays within $1\,\%$ at FP64 for both narrow layouts (${ratio_of('fp64_r16'):.3f}$ and ${ratio_of('fp64_lp'):.3f}$), slips to ${ratio_of('fp32_lp'):.3f}$ for CSR-lp at FP32, and falls to ${ratio_of('fp16_r16'):.3f}$ for CSR-r16 and ${ratio_of('fp16_lp'):.3f}$ for CSR-lp at FP16.
The last column locates the cause, since $S/\hat S$ is exactly $R/R_{{\mathrm{{ref}}}}$.
The attained throughput that Eq.~\eqref{{eq:prediction}} assumes unchanged stays at a median ${throughput_of('fp64_i32'):.2f}$ to ${throughput_of('fp16_i32'):.2f}\,B$ for CSR-i32 at all three precisions, but falls from ${throughput_of('fp64_r16'):.2f}$ to ${throughput_of('fp16_r16'):.2f}\,B$ for CSR-r16 and from ${throughput_of('fp64_lp'):.2f}$ to ${throughput_of('fp16_lp'):.2f}\,B$ for CSR-lp as their traffic shrinks from about ten to about four bytes per nonzero.
Two of the conditions of Sec.~\ref{{sec:model}} weaken together at that point.
The kernel no longer saturates the memory system, which is the third limit of Sec.~\ref{{sec:model-limits}}, and the per-nonzero unpacking of CSR-lp, which does not shrink with the bytes, takes a larger share of a shorter time.
The deviation is therefore measured rather than unexplained, and it marks the boundary of the model's domain rather than an error inside it.

The two levers can be compared directly on all {len(records)} matrices, since FP64 CSR-r16 is exact on every one of them.
FP32 CSR-i32 moves a median of ${median_of('fp32_i32_bytes'):.2f}$ bytes per nonzero against ${median_of('fp64_r16_bytes'):.2f}$ for FP64 CSR-r16, and it is faster {faster_text}, by a median of ${statistics.median(lever_measured):.3f}\times$ (range {span(lever_measured)}) against a predicted ${statistics.median(lever_predicted):.3f}\times$ (median ratio ${statistics.median(lever_ratio):.3f}$), while dropping to FP16 instead gives ${statistics.median(half_lever):.3f}\times$.
Equivalently, the lossless 16-bit index delivers a median ${statistics.median(lever_share):.2f}$ of the speedup that halving the value precision would deliver, while keeping the result of the FP64 product bit for bit.

Fig.~\ref{{fig:performance}} states the same measurements in absolute terms, each bar against the memory roof $2B/q$ that its own traffic allows.
At FP64 CSR-i32, CSR-r16 and CSR-lp reach medians of {rates_of('fp64')}~GFLOP/s against roofs of {roofs_of('fp64')}.
At FP32 they reach {rates_of('fp32')} against {roofs_of('fp32')}, and at FP16 {rates_of('fp16')} against {roofs_of('fp16')}.
A narrower index raises the bar and its roof together, up to {statistics.median(record['fp16_r16_gflops'] for record in records):.0f}~GFLOP/s for FP16 CSR-r16 against the {statistics.median(record['fp64_i32_gflops'] for record in records):.0f} of FP64 CSR-i32, while the distance from each bar to its own roof widens as the traffic shrinks.

\begin{{figure}}[t]
  \centering
  \includegraphics[width=\columnwidth]{{figures/performance.pdf}}
  \caption{{Performance of every layout at the three value precisions, median over the {len(records)} matrices, with the interquartile range and, as a tick, the memory roof $2B/q$ of that configuration (cuSPARSE has no FP16 mode here).}}
  \label{{fig:performance}}
\end{{figure}}

\begin{{extended}}
\begin{{figure*}}[t]
  \centering
  \includegraphics[width=\textwidth]{{figures/precision_speedup.pdf}}
  \caption{{Measured against predicted speedup over FP64 CSR-i32, one point per matrix and configuration, on the {len(records)} matrices whose row spans all fit 16 bits.
  Colour is the value precision and the marker the layout, and the line is equality.
  \textbf{{A}} predicts with the byte count alone, where the FP16 layouts fall below the line because the model charges them no decoding.
  \textbf{{B}} adds the cost of the instructions the compiled kernel spends per nonzero, which pulls the same points onto the line without fitting anything.
  Hollow grey markers are the matrices the test of Eq.~\eqref{{eq:applicable}} excludes, where no byte count applies.}}
  \label{{fig:precision-speedup}}
\end{{figure*}}
\end{{extended}}
"""


def lane_table_lines(records):
    """CSR-lp by domain: the packed width it reaches, what it gains, how well the byte model predicts
    that gain, and the row-length imbalance that explains where the prediction fails."""
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{CSR-lp by domain, at FP64 on all " + str(len(records)) +
             r" matrices, showing medians of the packed bits per nonzero, of the measured speedup $S$ over CSR-i32, of"
             r" $S/\hat S$ and of the row-length imbalance. $n_{\mathrm{r16}}$ counts the matrices of that domain"
             r" on which CSR-r16 is also exact.}",
             r"\label{tab:results-lane}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"domain & matrices & $n_{\mathrm{r16}}$ & bits & $S$ & $S/\hat S$ & $\ell_{\max}/\ell$ \\",
             r"\midrule"]
    median_row = lambda group: (rf"{statistics.median(record['bits'] for record in group):.1f}"
                                rf" & {statistics.median(record['speedup'] for record in group):.3f}"
                                rf" & {statistics.median(record['speedup'] / record['predicted'] for record in group):.3f}"
                                rf" & {statistics.median(record['imbalance'] for record in group):.1f}")
    admissible_in = lambda group: sum(1 for record in group if record["bytes_r16"] is not None)
    for domain, label in DOMAIN_LABEL.items():
        group = [record for record in records if record["domain"] == domain]
        if not group:
            continue
        lines.append(rf"{label} & {len(group)} & {admissible_in(group)} & {median_row(group)} \\")
    lines.append(r"\midrule")
    lines.append(rf"all & {len(records)} & {admissible_in(records)} & {median_row(records)} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return lines


def lane_tex(records, admissible_count, total_matrices, imbalance_threshold=10.0):
    speedups = [record["speedup"] for record in records]
    accuracy = [record["speedup"] / record["predicted"] for record in records]
    within_five = sum(1 for value in accuracy if abs(value - 1.0) <= 0.05)
    faster = sum(1 for value in speedups if value > 1.0)
    over_cusparse = [record["over_cusparse"] for record in records if record["over_cusparse"]]
    beats_cusparse = sum(1 for value in over_cusparse if value > 1.0)
    failing = [record for record in records if record["speedup"] / record["predicted"] < 0.95]
    holding = [record for record in records if record["speedup"] / record["predicted"] >= 0.95]
    both = [record for record in records if record["against_r16"]]
    identical = sum(1 for record in records if record["bit_identical"])
    table_block = "\\begin{extended}\n" + "\n".join(lane_table_lines(records)) + "\n\\end{extended}"
    return rf"""
\subsection{{A layout that is exact on every matrix}}
\label{{sec:results-lane}}

CSR-r16 is exact on {admissible_count} of the {total_matrices} matrices, and the question is what a layout that is always exact costs.
CSR-lp of Sec.~\ref{{sec:model}} bit-packs the difference from the entry $T$ positions earlier in the row at the width of the widest difference in that row.
It spends a median of ${statistics.median(record['bits'] for record in records):.1f}$ bits per nonzero against the $32$ of CSR-i32, excludes no matrix, keeps the kernel shape of CSR-i32 and returns results bit-identical to it on all {identical} matrices.

At FP64 it is faster than CSR-i32 on {faster} of the {total_matrices} matrices, with a median speedup of ${statistics.median(speedups):.3f}$ against a predicted ${statistics.median(record['predicted'] for record in records):.3f}$, and faster than the better cuSPARSE algorithm on {beats_cusparse} of them (median ${statistics.median(over_cusparse):.3f}$).
\extonly{{Table~\ref{{tab:results-lane}} breaks this down by domain.}}

{table_block}

On the {len(both)} matrices where CSR-r16 also applies it is the better of the two, by a median of ${statistics.median(record['against_r16'] for record in both):.3f}\times$, because a width chosen per row beats a fixed two bytes, at ${statistics.median(record['bytes'] for record in both):.2f}$ bytes per nonzero against ${statistics.median(record['bytes_r16'] for record in both):.2f}$.

The byte model is less accurate here than for CSR-r16, and the failures have a single cause.
The ratio $S/\hat S$ is within $5\,\%$ on {within_five} of the {total_matrices} matrices, and the {len(failing)} that fall below $0.95$ have a median row-length imbalance $\ell_{{\max}}/\ell$ of ${statistics.median(record['imbalance'] for record in failing):.0f}$ against ${statistics.median(record['imbalance'] for record in holding):.1f}$ for the rest.
\extonly{{Fig.~\ref{{fig:lane-accuracy}} separates the two groups at $\ell_{{\max}}/\ell = {imbalance_threshold:.0f}$, and w}}\briefonly{{W}}here the rows are even the measurements \extonly{{sit on the identity line}}\briefonly{{match the prediction}}, and where a few rows dominate it, a part of the time does not scale with the traffic at all, which is the mechanism described in Sec.~\ref{{sec:model}} and not an error in the byte count.

\begin{{extended}}
\begin{{figure}}[t]
  \centering
  \includegraphics[width=\columnwidth]{{figures/lane_accuracy.pdf}}
  \caption{{Measured against predicted speedup of CSR-lp over CSR-i32 at FP64, one point per matrix, on all {total_matrices} matrices.
  Circles are the matrices whose longest row is at most {imbalance_threshold:.0f} times the mean and triangles the rest.}}
  \label{{fig:lane-accuracy}}
\end{{figure}}
\end{{extended}}
"""


def rcm_table_lines(records):
    """What reordering changes, by domain: admissibility of the fixed 16-bit offset, the packed width
    of CSR-lp, and the speed of CSR-i32 itself."""
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Reverse Cuthill--McKee by domain, on the " + str(len(records)) +
             r" matrices of tiers M, L and XL. $n_{\mathrm{r16}}$ counts the matrices on which CSR-r16 is exact, before"
             r" and after the permutation. The bits columns give the median packed width of CSR-lp per nonzero, before and after."
             r" $t_{\mathrm{orig}}/t_{\mathrm{rcm}}$ is the median effect of the permutation on CSR-i32 itself, which"
             r" moves no bytes and only changes the gather.}",
             r"\label{tab:results-rcm}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r" & & \multicolumn{2}{c}{$n_{\mathrm{r16}}$} & \multicolumn{2}{c}{bits} & \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"domain & matrices & orig. & RCM & orig. & RCM & $t_{\mathrm{orig}}/t_{\mathrm{rcm}}$ \\",
             r"\midrule"]
    for domain, label in DOMAIN_LABEL.items():
        group = [record for record in records if record["domain"] == domain]
        if not group:
            continue
        lines.append(rf"{label} & {len(group)} & {sum(1 for r in group if r['admissible_original'])}"
                     rf" & {sum(1 for r in group if r['admissible_rcm'])}"
                     rf" & {statistics.median(r['bits_original'] for r in group):.1f}"
                     rf" & {statistics.median(r['bits_rcm'] for r in group):.1f}"
                     rf" & {statistics.median(r['i32_gain'] for r in group):.2f} \\")
    lines.append(r"\midrule")
    lines.append(rf"all & {len(records)} & {sum(1 for r in records if r['admissible_original'])}"
                 rf" & {sum(1 for r in records if r['admissible_rcm'])}"
                 rf" & {statistics.median(r['bits_original'] for r in records):.1f}"
                 rf" & {statistics.median(r['bits_rcm'] for r in records):.1f}"
                 rf" & {statistics.median(r['i32_gain'] for r in records):.2f} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return lines


def rcm_tex(records, cost):
    table_block = "\n".join(rcm_table_lines(records))
    admissible_before = sum(1 for record in records if record["admissible_original"])
    admissible_after = sum(1 for record in records if record["admissible_rcm"])
    gained = [record for record in records if record["admissible_rcm"] and not record["admissible_original"]]
    lost = [record for record in records if record["admissible_original"] and not record["admissible_rcm"]]
    gains = [record["r16_speedup"] for record in gained]
    predicted = [record["r16_predicted"] for record in gained]
    end_to_end = [record["r16_end_to_end"] for record in gained]
    narrower = sum(1 for record in records if record["bits_rcm"] < record["bits_original"])
    widest = max(records, key=lambda record: record["bits_rcm"] - record["bits_original"])
    kernel_gain = [record["i32_gain"] for record in records]
    accuracy_before = [record["lp_accuracy_original"] for record in records]
    accuracy_after = [record["lp_accuracy_rcm"] for record in records]
    accuracy_r16 = [record["r16_accuracy_rcm"] for record in records if record["r16_accuracy_rcm"] is not None]
    cost_text = ("" if cost is None else
                 f" The permutation itself costs a median ${cost[0]:.1f}$~s on one core, and ${cost[1]:.0f}$~s on the"
                 f" largest graph, which a matrix reused across many products amortises and a matrix used once does not.")
    lost_text = ("" if not lost else
                 f", gaining {len(gained)} and losing "
                 + ("one, " if len(lost) == 1 else f"{len(lost)}, ")
                 + ", ".join(f"\\texttt{{{tex_escape(record['matrix'])}}}" for record in lost))
    return rf"""
\subsection{{Reordering}}
\label{{sec:results-rcm}}

Admissibility is a property of the ordering, not of the matrix, so we repeat the measurement after a symmetric Reverse Cuthill--McKee permutation $P A P^{{\top}}$ on the {len(records)} matrices of tiers M, L and XL, RCM being the heuristic that minimises the bandwidth $\max_{{k}} |r_k - c_k|$, which is the worst row span of Eq.~\eqref{{eq:rowspan}}.

\begin{{brief}}
It takes the matrices on which CSR-r16 is exact from {admissible_before} to {admissible_after} of the {len(records)}{lost_text}, and narrows CSR-lp from a median of ${statistics.median(record['bits_original'] for record in records):.1f}$ to ${statistics.median(record['bits_rcm'] for record in records):.1f}$ bits per nonzero.
On the {len(gained)} matrices it unlocks, CSR-r16 is ${statistics.median(gains):.3f}\times$ faster than CSR-i32 there, against a predicted ${statistics.median(predicted):.3f}\times$.
The permutation moves no bytes and changes only the gather, so it is a lever on $R$ and not on $q$.
It leaves CSR-i32 at a median ${statistics.median(kernel_gain):.3f}\times$, which puts a matrix and its permutation outside the domain of the model, while inside a fixed ordering $S/\hat S$ stays at ${statistics.median(accuracy_r16):.3f}$.
\end{{brief}}

\begin{{extended}}
It takes the admissible matrices from {admissible_before} to {admissible_after} of the {len(records)}{lost_text}, and narrows the packed width of CSR-lp from a median of ${statistics.median(record['bits_original'] for record in records):.1f}$ to ${statistics.median(record['bits_rcm'] for record in records):.1f}$ bits per nonzero, on {narrower} of the {len(records)} matrices.
{table_block}

On the {len(gained)} matrices it makes admissible, CSR-r16 is then ${statistics.median(gains):.3f}\times$ faster than CSR-i32 on the same reordered matrix, against a predicted ${statistics.median(predicted):.3f}\times$, and ${statistics.median(end_to_end):.3f}\times$ faster than CSR-i32 on the original one (range {span(end_to_end, 2)}).
The permutation is therefore worth the two bytes it unlocks, but not because it makes the kernel faster by itself.
Reordering moves no bytes and changes only the accesses to $x$, and on this device it is not a lever, leaving CSR-i32 at a median ${statistics.median(kernel_gain):.3f}\times$ with extremes ${min(kernel_gain):.2f}$ and ${max(kernel_gain):.2f}$.{cost_text}

Two things follow for the model.
First, comparing a matrix with its permutation is outside its domain, since the two runs issue different accesses to $x$ and the third condition of Sec.~\ref{{sec:model}} fails, and the spread just quoted is what that failure looks like.
Second, inside a fixed ordering the model is unaffected, with $S/\hat S$ at a median ${statistics.median(accuracy_r16):.3f}$ for CSR-r16 and ${statistics.median(accuracy_after):.3f}$ for CSR-lp after the permutation, against ${statistics.median(accuracy_before):.3f}$ for CSR-lp before it.
Reordering and index compression are thus separate levers, one on the attained throughput $R$ and one on the bytes $q$, and the accounting keeps them apart.

The gain is also not uniform, since RCM narrows CSR-lp where the matrix has a geometric structure to exploit and widens it where it has none, by up to ${widest['bits_rcm'] - widest['bits_original']:.1f}$ bits on \texttt{{{tex_escape(widest['matrix'])}}}.
Since the packed width is a median over rows rather than a maximum, an ordering that optimises locality rather than the worst span, such as a graph partitioning, is the natural thing to try next for CSR-lp, while for CSR-r16 the worst span is the binding constraint and RCM is the matching objective.
\end{{extended}}
"""


def summarize_regime(table, characterization, instructions):
    """Per configuration, the error of the byte model and of the byte model plus the decoding cost,
    on the matrices the applicability test of Sec.~\\ref{sec:model} declares in the memory-bound regime."""
    issue_rate = DEVICE.issue_rate()
    if not issue_rate:
        return []
    records = []
    for precision in ("fp64", "fp32", "fp16"):
        for layout in ("r16", "lp"):
            plain, extended = [], []
            for matrix, row in characterization.items():
                entries = table.get((matrix, precision), {})
                reference = table.get((matrix, "fp64"), {}).get("i32")
                if layout not in entries or reference is None or not is_applicable(row):
                    continue
                measured = ((float(reference["t_ms"]) / int(reference["nnz"]))
                            / (float(entries[layout]["t_ms"]) / int(entries[layout]["nnz"])))
                cost = lambda entry, name, prec: (float(entry["bytes_per_nnz"]) / DEVICE.read_bw_gbs
                                                  + instructions.get((name, prec, int(entry["best_tpr"])), 0.0)
                                                  / issue_rate * 1e9)
                bytes_only = float(reference["bytes_per_nnz"]) / float(entries[layout]["bytes_per_nnz"])
                with_work = cost(reference, "i32", "fp64") / cost(entries[layout], layout, precision)
                plain.append(abs(bytes_only / measured - 1.0))
                extended.append(abs(with_work / measured - 1.0))
            if not plain:
                continue
            records.append({"precision": precision, "layout": layout, "count": len(plain),
                            "plain_median": statistics.median(plain),
                            "plain_within": sum(1 for value in plain if value <= 0.05),
                            "extended_median": statistics.median(extended),
                            "extended_within": sum(1 for value in extended if value <= 0.05)})
    return records


def regime_table_lines(records, instructions):
    widths = ", ".join(f"{instructions.get((layout, 'fp64', 32), 0.0):.1f} for CSR-{layout}"
                       for layout in ("i32", "r16", "lp")) + " at FP64"
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{The byte model against the byte model plus the decoding cost of Eq.~\eqref{eq:two-term},"
             r" on the matrices the test of Eq.~\eqref{eq:applicable} places in the memory-bound regime."
             r" Speedups are over FP64 CSR-i32. Neither column fits anything, since $B$ and the issue rate"
             r" are machine figures and the instructions per nonzero are read from the compiled kernels,"
             r" where at 32 threads per row they are " + widths + r".}",
             r"\label{tab:results-regime}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lrrrrr}", r"\toprule",
             r" & & \multicolumn{2}{c}{bytes only} & \multicolumn{2}{c}{with decoding} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"configuration & $n$ & median & $\le 5\,\%$ & median & $\le 5\,\%$ \\",
             r"\midrule"]
    for record in records:
        lines.append(rf"{record['precision'].upper()} CSR-{record['layout']} & {record['count']}"
                     rf" & {100 * record['plain_median']:.1f}\,\% & {record['plain_within']}"
                     rf" & {100 * record['extended_median']:.1f}\,\% & {record['extended_within']} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return lines


def regime_tex(records, characterization, instructions):
    flagged = [name for name, row in characterization.items() if not is_applicable(row)]
    table_block = "\n".join(regime_table_lines(records, instructions))
    find = lambda precision, layout: next(r for r in records if r["precision"] == precision and r["layout"] == layout)
    return rf"""
\subsection{{Where the prediction applies, and what decoding costs}}
\label{{sec:results-regime}}

Sec.~\ref{{sec:results-fp32}} found the byte model failing as the traffic per nonzero shrinks.
Two quantities that cost no measurement explain most of it.

\begin{{brief}}
The test places {len(flagged)} of the {len(characterization)} matrices outside the memory-bound regime, and the measurement agrees, since CSR-i32 itself runs at between $3$ and $23\,\%$ of its own memory bound there against a median $84\,\%$ elsewhere, which is why no ratio of byte counts can be right on them.
On the {find('fp16', 'lp')['count']} it admits, charging the decoding instructions of the compiled kernel halves the remaining error at FP16, from ${100 * find('fp16', 'lp')['plain_median']:.1f}\,\%$ to ${100 * find('fp16', 'lp')['extended_median']:.1f}\,\%$ for CSR-lp and from ${100 * find('fp16', 'r16')['plain_median']:.1f}\,\%$ to ${100 * find('fp16', 'r16')['extended_median']:.1f}\,\%$ for CSR-r16.
\end{{brief}}

\begin{{extended}}
{table_block}

The test of Eq.~\eqref{{eq:applicable}} places {len(flagged)} of the {len(characterization)} matrices outside the memory-bound regime.
They are the ones whose longest row outlasts every other row spread over the machine, and the measurement agrees, since CSR-i32 itself reaches between $3$ and $23\,\%$ of its own memory bound there against a median $84\,\%$ elsewhere.
On those matrices the byte model is not inaccurate but inapplicable, with a median error above $60\,\%$ for CSR-lp, and no ratio of byte counts can be right when CSR-lp runs twice as slowly as CSR-i32 while reading fewer bytes.
The ratio follows the memory-level parallelism instead, which the unroll factor of the compiled loop sets, $16$ for CSR-i32 against $4$ for CSR-lp.

On the {find('fp16', 'lp')['count']} matrices the test admits, Table~\ref{{tab:results-regime}} compares the two forms.
The decoding cost helps exactly where the traffic per nonzero is small, taking CSR-lp at FP16 from ${100 * find('fp16', 'lp')['plain_median']:.1f}\,\%$ to ${100 * find('fp16', 'lp')['extended_median']:.1f}\,\%$ and CSR-r16 from ${100 * find('fp16', 'r16')['plain_median']:.1f}\,\%$ to ${100 * find('fp16', 'r16')['extended_median']:.1f}\,\%$, and it leaves FP64 where it was, since there a nonzero carries ten times the bytes and the same handful of instructions.
The one configuration it does not improve is CSR-lp at FP64, where the byte model was already within ${100 * find('fp64', 'lp')['plain_median']:.1f}\,\%$ and the added term overshoots.
\end{{extended}}
"""


def device_constants_tex(records):
    """The machine numbers the hand-written sections quote, as macros.

    Background and Model name the bandwidth, the roofs and the ridge point in prose. Written out as
    digits they belong to whichever machine was current when the sentence was typed, and they go
    stale silently the day the paper reports another one. Written as macros they follow device.txt.

    The arithmetic intensity is that of FP64 CSR-i32, which is the configuration the ridge point is
    compared against, and it comes from the measured byte count rather than from a formula, so the
    ratio in the prose is the one the campaign actually saw.
    """
    # memory_i32 is the working set in MiB, so the bytes per nonzero need the factor back
    intensity = 2.0 / statistics.median(record["memory_i32"] * MIB / record["nnz"] for record in records)
    ridge = DEVICE.peak_gflops["fp64"] / DEVICE.read_bw_gbs
    macros = [
        ("devName", DEVICE.name),
        ("devArch", DEVICE.arch),
        ("devDescription", DEVICE.description()),
        ("devSMs", f"{int(DEVICE.multiprocessors)}"),
        ("devLtwo", f"{int(DEVICE.l2_mib)}"),
        ("devResident", f"{int(DEVICE.resident_threads())}"),
        ("devBtheo", f"{DEVICE.theoretical_bw_gbs:.1f}"),
        ("devBread", f"{DEVICE.read_bw_gbs:.1f}"),
        ("devBtriad", f"{DEVICE.triad_bw_gbs:.1f}"),
        ("devBreadPct", f"{100 * DEVICE.read_bw_gbs / DEVICE.theoretical_bw_gbs:.0f}"),
        ("devBtriadPct", f"{100 * DEVICE.triad_bw_gbs / DEVICE.theoretical_bw_gbs:.0f}"),
        ("devPeakFPsixtyfour", f"{DEVICE.peak_gflops['fp64']:.1f}"),
        ("devRidge", f"{ridge:.2f}"),
        ("devIntensity", f"{intensity:.3f}"),
        ("devRidgeRatio", f"{ridge / intensity:.0f}"),
        ("devCuda", DEVICE.cuda or ""),
        ("devCusparse", DEVICE.cusparse or ""),
    ]
    header = [
        "% Generated by analysis/make_i16_results.py from results/<device>/device.txt. Do not edit.",
        "% Input it from the preamble and quote the macros instead of digits, so that the prose",
        "% follows the machine the paper reports rather than the one it happened to be written on.",
    ]
    body = [chr(92) + "newcommand{" + chr(92) + name + "}{" + value + "}" for name, value in macros]
    return chr(10).join(header + body) + chr(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=ROOT / "results" / DEVICE_DIR / "csr_layouts.csv")
    parser.add_argument("--device", type=Path, default=None,
                        help="device.txt of the machine (default: next to the CSV)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    # reordered.sbatch writes csr_layouts_rcm.csv; the older rcm_campaign.sbatch writes
    # rcm_campaign.csv and measures both orderings into it, so either name is accepted
    reordered = next((ROOT / "results" / DEVICE_DIR / name
                      for name in ("csr_layouts_rcm.csv", "rcm_campaign.csv")
                      if (ROOT / "results" / DEVICE_DIR / name).exists()),
                     ROOT / "results" / DEVICE_DIR / "csr_layouts_rcm.csv")
    parser.add_argument("--rcm-csv", type=Path, default=reordered)
    parser.add_argument("--census-rcm-csv", type=Path, default=ROOT / "data" / "results" / "a100" / "csr_layouts_rcm.csv",
                        help="a reordered campaign covering the whole matrix set, for the admissibility "
                             "figure only. Admissibility is a property of the matrix and the permutation "
                             "rather than of the device, and the timing campaign of the reference machine "
                             "covers only the larger tiers, so the census is read off whichever run is complete.")
    parser.add_argument("--characterization", type=Path, default=None,
                        help="structural pass to read (default: the first complete one of "
                             + ", ".join(CHARACTERIZATION_DIRS) + ")")
    parser.add_argument("--table-input", default="sections/tables/results_i16_table",
                        help="path results.tex uses to \\input the per-matrix table (relative to the paper)")
    parser.add_argument("--reps", type=int, default=20, help="timing samples of the campaign, for the prose")
    parser.add_argument("--warmup", type=int, default=3, help="warm-up products of the campaign, for the prose")
    parser.add_argument("--min-sample-ms", type=float, default=10.0,
                        help="minimum duration of a timing sample in the campaign, for the prose")
    args = parser.parse_args()

    global DEVICE
    DEVICE = load_device(args.csv, args.device)
    table = load_measurements(args.csv)
    characterization = load_characterization(args.characterization)
    fp64_records = summarize_fp64(table, characterization)
    precision_records = summarize_precisions(table, characterization)
    lane_records = summarize_lane(table, characterization)
    rcm_records = summarize_rcm(load_rcm_measurements(args.rcm_csv, args.csv), characterization)
    instructions = load_instruction_counts(args.csv.with_name("instructions.csv"))
    regime_records = summarize_regime(table, characterization, instructions)
    identical, cusparse_deviation = measurement_checks(table)
    warn_if_cusparse_fp16(args.csv)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    census = rcm_admissible(args.census_rcm_csv)
    if census and not set(characterization) <= census[0]:
        missing = len(set(characterization) - census[0])
        print(f"  {args.census_rcm_csv.name} reorders {len(census[0])} matrices and the set has "
              f"{len(characterization)}, so {missing} would be missing: the admissibility figure keeps "
              "the original ordering alone")
        census = None
    counts = admissibility_figure(characterization, figures_dir, census)
    census_summary = None
    if census:
        before = {name for name, row in characterization.items() if float(row["rows_fitting_16bit"]) == 1.0}
        agreement = census_agreement(args.rcm_csv, args.census_rcm_csv)
        if agreement and agreement[0] != agreement[1]:
            print(f"  the two reordered campaigns disagree on {agreement[0] - agreement[1]} of the "
                  f"{agreement[0]} matrices both reordered, so the census is not device independent")
        census_summary = {"before": len(before), "after": len(census[1]),
                          "gained": len(census[1] - before), "lost": len(before - census[1]),
                          "agreed": agreement[1] if agreement else 0}
    i16_speedup_figure(fp64_records, figures_dir)
    roofline_panels(fp64_records, table, characterization, figures_dir)
    precision_speedup_figure(precision_records, figures_dir, instructions)
    performance_figure(precision_records, figures_dir)
    lane_accuracy_figure(lane_records, figures_dir)
    # written where the paper reads them, so no copy stands between a regeneration and the next build
    tables_dir = args.output_dir / "sections" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / "results_i16_table.tex").write_text(i16_table_tex(fp64_records))
    (args.output_dir / "sections" / "device_constants.tex").write_text(device_constants_tex(fp64_records))
    (args.output_dir / "sections" / "results.tex").write_text(
        intro_tex() + setup_tex(len(fp64_records), sum(1 for record in fp64_records
                                                       if record["absolute_speedup"] is not None),
                                identical, cusparse_deviation, (args.reps, args.warmup, args.min_sample_ms))
        + i16_tex(fp64_records, counts, len(characterization), args.table_input, census_summary)
        + fp32_tex(precision_records)
        + lane_tex(lane_records, len(fp64_records), len(characterization))
        + rcm_tex(rcm_records, ordering_seconds(args.rcm_csv.with_suffix(".log")))
        + regime_tex(regime_records, characterization, instructions))
    print(f"device {DEVICE.name} from {DEVICE.path}, read roof {DEVICE.read_bw_gbs:.1f} GB/s; "
          f"{len(fp64_records)} matrices at FP64, {len(precision_records)} with FP32; "
          f"bit-identical: { {precision: all(flags) for precision, flags in identical.items()} }; "
          f"cuSPARSE worst relative deviation: {cusparse_deviation}")


if __name__ == "__main__":
    main()
