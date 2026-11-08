#!/usr/bin/env python3
"""make_results.py — Results section on the RCM-ordered matrices: admissibility, the gain of each
layout, the bound of Eq. sbound, what the rate ratio consists of and the cost of an implementation.

Every matrix is measured after a symmetric Reverse Cuthill-McKee permutation, which is what makes
CSR-r16 exact on 44 of the 74 matrices rather than 26, and every comparison is made inside that one
ordering.

Inputs
  --csv (default data/results/a100/csr_layouts_rcm.csv), the reordered campaign
  --original-csv (default csr_layouts.csv next to it), read only to count the admissible matrices
                 of the original ordering
  --element-major-csv (default previous/csr_layouts_rcm.csv next to it), the same campaign with the
                 element-major packing of CSR-lp, for the implementation paragraph
  --device (default the device.txt next to the CSV)
  --rate-study (default data/results/a100/k_study_rcm), the Nsight Compute exports of
                 k_study.sbatch, grouped by --rate-list, with the counted bytes and the T of CSR-i32
                 read from --rate-csv; --rate-ordering original reads the earlier k_study,
                 profiled before the reordering, against the campaign in the original ordering
Outputs, under --output-dir (default output/)
  sections/results.tex   the whole Results section, every number computed here
  figures/bound.pdf      measured speedup against the bound of Eq. sbound, every layout and precision
  figures/roofline.pdf   the two Roofline panels of the Model section, on the reordered matrices

One sentence per line in the generated LaTeX; parentheses instead of em-dashes, no colons or
semicolons in the prose.
"""
import argparse
import math
import statistics
from pathlib import Path

import numpy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import make_i16_results as shared  # noqa: E402
from device_profile import load_device  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PRECISIONS = (("fp64", "FP64", 8), ("fp32", "FP32", 4), ("fp16", "FP16", 2))
LAYOUTS = (("i16", "CSR-i16"), ("r16", "CSR-r16"), ("lp", "CSR-lp"))
LAYOUT_MARKER = {"i16": "D", "r16": "s", "lp": "^"}
ROW_POINTER_BYTES = 4.0
REFERENCE_INDEX_BYTES = 4.0
SECTOR_BYTES = 32.0
# longest row over the mean row length above which a matrix counts as uneven
IMBALANCE_LIMIT = 10.0
PREDICTION_TOLERANCE = 0.10
COUNT_TOLERANCE = 0.07
BAND_LAYOUTS = (("r16", "CSR-r16"), ("lp", "CSR-lp"))


# ---------------------------------------------------------------------------
# Per-matrix quantities
# ---------------------------------------------------------------------------

def counted_rate(entry):
    """K of Eq. kdef in GB/s, the counted bytes over the measured time of the whole kernel."""
    return float(entry["bytes_total"]) / (float(entry["t_ms"]) * 1e-3) / 1e9


def byte_ratio_ceiling(entry, value_bytes):
    """r_max of Eq. rmax, from the dimensions of the matrix alone."""
    rows, cols, nonzeros = int(entry["rows"]), int(entry["cols"]), int(entry["nnz"])
    auxiliary = ((rows + 1) * ROW_POINTER_BYTES + (cols + rows) * value_bytes) / nonzeros
    return (value_bytes + REFERENCE_INDEX_BYTES + auxiliary) / (value_bytes + auxiliary)


def gather_band(reference, candidate, value_bytes):
    """Eq. band, the speedup with x read once and with a 32-byte sector charged for every gather,
    returned as (low, high)."""
    reference_bytes = float(reference["bytes_per_nnz"])
    candidate_bytes = float(candidate["bytes_per_nnz"])
    vector_bytes = int(reference["cols"]) * value_bytes / int(reference["nnz"])
    once = reference_bytes / candidate_bytes
    sector = (reference_bytes - vector_bytes + SECTOR_BYTES) / (candidate_bytes - vector_bytes + SECTOR_BYTES)
    return min(once, sector), max(once, sector)


def band_position(measured):
    """0 at the sector end of the band, 1 at the end with x read once."""
    low, high = measured["band"]
    return (measured["speedup"] - low) / (high - low) if high > low else float("nan")


def summarize(table, device, characterization):
    """One record per matrix and precision with the reference, the bound and every admissible layout."""
    l2_bytes = device.l2_mib * shared.MIB
    records = []
    for (matrix, precision), entries in table.items():
        if "i32" not in entries:
            continue
        value_bytes = dict((key, size) for key, _, size in PRECISIONS)[precision]
        reference = entries["i32"]
        efficiency = counted_rate(reference) / device.read_bw_gbs
        ceiling = byte_ratio_ceiling(reference, value_bytes)
        record = {
            "matrix": matrix,
            "precision": precision,
            "row_length": float(characterization[matrix]["row_length_mean"]),
            "imbalance": int(characterization[matrix]["row_length_max"]) / float(characterization[matrix]["row_length_mean"]),
            "efficiency": efficiency,
            "r_max": ceiling,
            "bound": ceiling / efficiency,
            "resident": float(reference["bytes_total"]) < l2_bytes,
            "working_set": float(reference["bytes_total"]) / l2_bytes,
            "time_ms": float(reference["t_ms"]),
            "bytes": float(reference["bytes_total"]),
            "q": float(reference["bytes_per_nnz"]),
            "rate": counted_rate(reference),
            "cusparse_over_i32": (float(shared.best_cusparse(entries)["t_ms"]) / float(reference["t_ms"])
                                  if any(name in entries for name in shared.CUSPARSE) else None),
            "layouts": {},
        }
        for layout, _ in LAYOUTS:
            if layout not in entries:
                continue
            candidate = entries[layout]
            record["layouts"][layout] = {
                "speedup": float(reference["t_ms"]) / float(candidate["t_ms"]),
                "byte_ratio": float(reference["bytes_total"]) / float(candidate["bytes_total"]),
                "bits": 8.0 * float(candidate["bytes_columns"]) / int(candidate["nnz"]),
                "q": float(candidate["bytes_per_nnz"]),
                "rate": counted_rate(candidate),
                "layout_bound": (float(reference["bytes_total"]) / float(candidate["bytes_total"])) / efficiency,
                "band": gather_band(reference, candidate, value_bytes),
                "over_cusparse": (float(shared.best_cusparse(entries)["t_ms"]) / float(candidate["t_ms"])
                                  if any(name in entries for name in shared.CUSPARSE) else None),
            }
        records.append(record)
    return records


def of_precision(records, precision):
    return [record for record in records if record["precision"] == precision]


def with_layout(records, layout):
    return [record for record in records if layout in record["layouts"]]


def admissible_original(csv_path):
    """Matrices on which CSR-r16 is exact in the original ordering, at FP64."""
    return {row["matrix"] for row in shared.read_rows(csv_path)
            if row["prec"] == "fp64" and row["status"] == "ok" and row["row_encoding"] == "cumulative"
            and row["kernel"] == "custom" and shared.label_of(row) == "r16"}


def lane_packed_gain(earlier_csv, table):
    """Median time of CSR-lp in an earlier campaign over its time in this one, per precision, and the
    largest relative change of the packed size between the two."""
    if earlier_csv is None or not earlier_csv.exists():
        return None
    earlier = shared.load_measurements(earlier_csv)
    gains = {}
    size_change = 0.0
    for (matrix, precision), entries in table.items():
        before = earlier.get((matrix, precision), {}).get("lp")
        if before is None or "lp" not in entries:
            continue
        gains.setdefault(precision, []).append(float(before["t_ms"]) / float(entries["lp"]["t_ms"]))
        size_change = max(size_change, abs(float(entries["lp"]["bytes_total"]) / float(before["bytes_total"]) - 1.0))
    return {precision: statistics.median(values) for precision, values in gains.items()}, size_change


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def bound_figure(records, figures_dir):
    """Every measured speedup, divided by r_max, against the efficiency of the reference on the same
    matrix. Eq. sbound reads S/r_max <= 1/eta_ref, a curve that depends on the reference alone, and
    nothing may lie above it unless the matrix is served from the L2, which the hollow markers show."""
    figure, axis = plt.subplots(figsize=(3.4, 2.8))
    efficiencies, fractions = [], []
    for precision, _, _ in PRECISIONS:
        colour = shared.PRECISION_COLOUR[precision]
        for record in of_precision(records, precision):
            for layout, measured in record["layouts"].items():
                efficiencies.append(record["efficiency"])
                fractions.append(measured["speedup"] / record["r_max"])
                axis.scatter([record["efficiency"]], [fractions[-1]], s=10, marker=LAYOUT_MARKER[layout],
                             facecolor="none" if record["resident"] else colour, edgecolors=colour,
                             linewidths=0.6, alpha=0.85, zorder=3)
    left, right = 0.8 * min(efficiencies), 1.1 * max(efficiencies)
    curve = [left * (right / left) ** (index / 200) for index in range(201)]
    axis.plot(curve, [1.0 / value for value in curve], color=shared.INK_SECONDARY, linewidth=1.1, zorder=2)
    axis.axvline(1.0, color=shared.GRID, linewidth=0.8, zorder=1)
    axis.axhline(1.0, color=shared.INK_SECONDARY, linewidth=0.8, linestyle=(0, (4, 2)), zorder=2)
    axis.annotate(r"$S = r_{\max}$", xy=(1.1 * left, 1.06), color=shared.INK_SECONDARY, fontsize=7)
    axis.annotate(r"bound, $S / r_{\max} = B / K_{\mathrm{ref}}$", xy=(0.4, 2.5), xytext=(0.012, 2.3),
                  color=shared.INK_SECONDARY, fontsize=7, va="center",
                  arrowprops={"arrowstyle": "-", "color": shared.INK_SECONDARY, "linewidth": 0.6})
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(left, right)
    axis.set_ylim(0.8 * min(fractions), 4.0)
    axis.set_yticks([0.25, 0.5, 1.0, 2.0, 4.0])
    axis.set_yticklabels(["0.25", "0.5", "1", "2", "4"])
    axis.yaxis.set_minor_formatter(plt.NullFormatter())
    axis.set_xlabel(r"$K_{\mathrm{ref}} / B$ of the reference")
    axis.set_ylabel(r"$S / r_{\max}$")
    axis.grid(True, which="major", color=shared.GRID, linewidth=0.5)
    shared.style_axis(axis)
    handles = [Line2D([], [], linestyle="none", marker="o", markersize=3.5, color=shared.PRECISION_COLOUR[precision],
                      label=label) for precision, label, _ in PRECISIONS]
    handles += [Line2D([], [], linestyle="none", marker=LAYOUT_MARKER[layout], markersize=3.5,
                       markerfacecolor=shared.INK_SECONDARY, markeredgecolor=shared.INK_SECONDARY, label=label)
                for layout, label in LAYOUTS]
    handles.append(Line2D([], [], linestyle="none", marker="o", markersize=3.5, markerfacecolor="none",
                          markeredgecolor=shared.INK_SECONDARY, label="fits in L2"))
    figure.legend(handles=handles, loc="upper center", frameon=False, fontsize=6, ncol=4, handletextpad=0.2,
                  columnspacing=0.9)
    figure.tight_layout(pad=0.3, rect=(0.0, 0.0, 1.0, 0.86))
    shared.save_figure(figure, "bound", figures_dir)
    plt.close(figure)


def band_subset(records, precision, layout):
    """The matrices the band is meant for, served from the DRAM and with balanced rows, sorted by the
    upper end of their band."""
    group = [record for record in with_layout(of_precision(records, precision), layout)
             if not record["resident"] and record["imbalance"] <= IMBALANCE_LIMIT]
    return sorted(group, key=lambda record: record["layouts"][layout]["band"][1])


def inside_band(measured):
    low, high = measured["band"]
    return low <= measured["speedup"] <= high


def band_figure(records, figures_dir):
    """The band of Eq. band of every matrix as a grey bar on the speedup axis and the measured speedup
    as a dot, a cross when it falls outside, one panel per layout and precision."""
    figure, axes = plt.subplots(len(BAND_LAYOUTS), len(PRECISIONS), figsize=(7.0, 3.6), sharey=True)
    for row, (layout, label) in enumerate(BAND_LAYOUTS):
        for column, (precision, _, _) in enumerate(PRECISIONS):
            axis = axes[row][column]
            colour = shared.PRECISION_COLOUR[precision]
            group = band_subset(records, precision, layout)
            for index, record in enumerate(group):
                measured = record["layouts"][layout]
                axis.plot([index, index], list(measured["band"]), color=shared.NEUTRAL_FILL, linewidth=3,
                          solid_capstyle="butt", zorder=1)
                inside = inside_band(measured)
                axis.scatter([index], [measured["speedup"]], s=8, marker="o" if inside else "x", linewidths=0.8,
                             color=colour if inside else shared.INK_PRIMARY, zorder=3)
            axis.axhline(1.0, color=shared.INK_SECONDARY, linewidth=0.6, linestyle=(0, (4, 2)))
            inside_count = sum(1 for record in group if inside_band(record["layouts"][layout]))
            axis.set_title(f"{precision.upper()} {label}, {inside_count}/{len(group)} inside", fontsize=7)
            axis.set_xticks([])
            shared.style_axis(axis)
            if column == 0:
                axis.set_ylabel("speedup")
            if row == len(BAND_LAYOUTS) - 1:
                axis.set_xlabel(r"matrices sorted by $q_{\mathrm{ref}}/q$", fontsize=7)
    axes[0][0].set_ylim(0.75, 1.75)
    figure.tight_layout(pad=0.3)
    shared.save_figure(figure, "band", figures_dir)
    plt.close(figure)


# offsets in points of the count of slower matrices, one height per layout so the three never overlap
LABEL_OFFSET = {"i16": (3, 16), "r16": (3, 8), "lp": (3, -9)}
PERFORMANCE_LAYOUTS = (("i32", "CSR-i32"), ("r16", "CSR-r16"), ("lp", "CSR-lp"), ("cusparse", "cuSPARSE"))


def load_thread_samples(path):
    """(matrix, precision, layout) -> {threads per row: row} from the per-threads-per-row CSV of a
    campaign. A resumed campaign repeats the matrix it was cut off in, and the later rows win."""
    samples = {}
    for row in shared.read_rows(path):
        samples.setdefault((row["matrix"], row["prec"], row["layout"]), {})[int(row["tpr"])] = row
    return samples


def at_reference_threads(table, samples):
    """The table of the campaign with every layout timed at the threads per row that is fastest for
    CSR-i32 on the same matrix and precision, the same assignment of threads to rows the first
    condition of the model asks for. CSR-i32 itself and cuSPARSE are unchanged. Returns that table and
    the time of each layout at its own best threads per row over its time at the reference's, per
    precision and layout, which is what letting a layout choose for itself would add."""
    equal = {}
    own_gain = {}
    for (matrix, precision), entries in table.items():
        copied = dict(entries)
        reference = entries.get("i32")
        if reference is not None:
            reference_threads = int(reference["best_tpr"])
            for layout, row in entries.items():
                timed = samples.get((matrix, precision, layout), {})
                if layout in ("i32",) or layout.startswith("cusparse") or reference_threads not in timed:
                    continue
                sample = timed[reference_threads]
                adjusted = dict(row)
                adjusted["t_ms"] = sample["t_ms"]
                adjusted["best_tpr"] = str(reference_threads)
                # CSR-lp packs a different stream for each threads per row, so its bytes move with it,
                # and only its column stream changes between candidates
                adjusted["bytes_columns"] = str(float(row["bytes_columns"]) + float(sample["bytes_total"])
                                                - float(row["bytes_total"]))
                adjusted["bytes_total"] = sample["bytes_total"]
                adjusted["bytes_per_nnz"] = str(float(sample["bytes_total"]) / int(row["nnz"]))
                adjusted["gflops"] = str(2.0 * int(row["nnz"]) / (float(sample["t_ms"]) * 1e-3) / 1e9)
                adjusted["max_abs_diff"] = sample["max_abs_diff"]
                copied[layout] = adjusted
                own_gain.setdefault((precision, layout), []).append(float(sample["t_ms"]) / float(row["t_ms"]))
        equal[(matrix, precision)] = copied
    # most matrices have one fastest threads per row for every layout, so the gain is reported over the
    # matrices where it differs, together with how many they are
    summary = {}
    for key, values in own_gain.items():
        changed = [value for value in values if value > 1.0]
        summary[key] = (len(changed), len(values), statistics.median(changed) if changed else 1.0)
    return equal, summary


def above_the_roof(table, device):
    """Matrices on which CSR-i32 or CSR-r16 retires its counted bytes faster than the DRAM can supply
    them at any precision, which only a matrix served from the L2 can do. The Roofline figure leaves
    them out, since its roof is that of the DRAM."""
    matrices = set()
    for (matrix, _), entries in table.items():
        for layout in ("i32", "r16"):
            entry = entries.get(layout)
            if entry is not None and counted_rate(entry) > device.read_bw_gbs:
                matrices.add(matrix)
    return matrices


def performance_set(table, device):
    """Matrices on which every layout of the figure is exact and whose working set at FP64 exceeds the
    L2, so that the DRAM roof applies to all of them."""
    l2_bytes = device.l2_mib * shared.MIB
    return sorted(matrix for (matrix, precision), entries in table.items()
                  if precision == "fp64" and "i32" in entries and "r16" in entries and "lp" in entries
                  and float(entries["i32"]["bytes_total"]) >= l2_bytes)


def performance_summary(table, device):
    """(precision, layout) -> (median GFLOP/s, first quartile, third quartile, roof 2B/q at the median q)."""
    matrices = performance_set(table, device)
    summary = {}
    for precision, _, _ in PRECISIONS:
        for layout, _ in PERFORMANCE_LAYOUTS:
            rates, counted = [], []
            for matrix in matrices:
                entries = table.get((matrix, precision), {})
                entry = (shared.best_cusparse(entries) if layout == "cusparse"
                         and any(name in entries for name in shared.CUSPARSE) else entries.get(layout))
                if entry is None:
                    continue
                rates.append(float(entry["gflops"]))
                counted.append(float(entries["i32" if layout == "cusparse" else layout]["bytes_per_nnz"]))
            if len(rates) == len(matrices):
                ordered = sorted(rates)
                summary[(precision, layout)] = (median(rates), ordered[len(ordered) // 4],
                                                ordered[(3 * len(ordered)) // 4],
                                                2.0 * device.read_bw_gbs / median(counted))
    return matrices, summary


def performance_figure(table, device, figures_dir):
    """Absolute performance of every layout at every value precision on the matrices of performance_set,
    median with the interquartile range, each bar against the memory roof 2B/q of its own traffic."""
    _, summary = performance_summary(table, device)
    width = 0.26
    figure, axis = plt.subplots(figsize=(3.4, 2.3))
    for offset, (precision, label, _) in enumerate(PRECISIONS):
        for position, (layout, _) in enumerate(PERFORMANCE_LAYOUTS):
            if (precision, layout) not in summary:
                continue
            middle, low, high, roof = summary[(precision, layout)]
            centre = position + (offset - 1) * width
            axis.bar([centre], [middle], width, color=shared.PRECISION_COLOUR[precision],
                     label=label if position == 0 else None, zorder=2)
            axis.errorbar([centre], [middle], yerr=[[middle - low], [high - middle]], fmt="none",
                          ecolor=shared.INK_SECONDARY, elinewidth=0.8, capsize=1.6, zorder=3)
            axis.plot([centre - width / 2, centre + width / 2], [roof, roof], color=shared.INK_PRIMARY,
                      linewidth=1.1, zorder=4)
    axis.plot([], [], color=shared.INK_PRIMARY, linewidth=1.1, label=r"memory roof $2B/q$")
    axis.set_xlim(-0.55, len(PERFORMANCE_LAYOUTS) - 0.45)
    axis.set_xticks(range(len(PERFORMANCE_LAYOUTS)))
    axis.set_xticklabels([label for _, label in PERFORMANCE_LAYOUTS], fontsize=7)
    axis.set_ylabel("performance [GFLOP/s]")
    axis.yaxis.grid(True, color=shared.GRID, linewidth=0.5)
    shared.style_axis(axis)
    figure.legend(loc="upper center", frameon=False, fontsize=6, ncol=4, handlelength=1.4, columnspacing=0.9,
                  handletextpad=0.4)
    figure.tight_layout(pad=0.3, rect=(0.0, 0.0, 1.0, 0.9))
    shared.save_figure(figure, "performance", figures_dir)
    plt.close(figure)


def speedup_figure(records, figures_dir):
    """Speedup of every layout over CSR-i32, sorted, one panel per value precision, with the number of
    matrices on which each layout is slower marked where its curve crosses one. r_max is not drawn: it
    bounds the byte ratio, not the speedup, which exceeds it where a matrix is served from the L2."""
    colours = {"i16": shared.SERIES_AQUA, "r16": shared.SERIES_BLUE, "lp": shared.SERIES_ORANGE}
    figure, axes = plt.subplots(1, 3, figsize=(7.0, 2.0), sharey=True)
    for axis, (precision, label, _) in zip(axes, PRECISIONS):
        group = of_precision(records, precision)
        axis.axhline(1.0, color=shared.INK_SECONDARY, linewidth=0.8, zorder=1)
        for layout, layout_label in LAYOUTS:
            speedups = sorted(record["layouts"][layout]["speedup"] for record in with_layout(group, layout))
            if not speedups:
                continue
            fractions = [(index + 0.5) / len(speedups) for index in range(len(speedups))]
            axis.plot(fractions, speedups, color=colours[layout], linewidth=1.4, marker=".", markersize=3,
                      label=f"{layout_label} ({len(speedups)})" if precision == "fp64" else None, zorder=3)
            # where the curve crosses one, the number of matrices on which the layout is slower
            slower = sum(1 for value in speedups if value < 1.0)
            axis.plot([slower / len(speedups)], [1.0], marker="|", markersize=7, color=colours[layout], zorder=4)
            axis.annotate(f"{slower}/{len(speedups)}", xy=(slower / len(speedups), 1.0),
                          xytext=LABEL_OFFSET[layout], textcoords="offset points", fontsize=6,
                          color=colours[layout], ha="left", va="center")
        axis.set_xlim(0.0, 1.0)
        axis.set_xticks([])
        axis.set_xlabel("matrices sorted by speedup")
        axis.set_title(label, fontsize=8, loc="left", color=shared.INK_PRIMARY)
        axis.yaxis.grid(True, color=shared.GRID, linewidth=0.5)
        shared.style_axis(axis)
    axes[0].set_ylabel("speedup over CSR-i32")
    axes[0].set_ylim(0.3, 2.0)
    axes[0].set_yticks([0.5, 1.0, 1.5, 2.0])
    figure.legend(loc="upper center", frameon=False, fontsize=7, ncol=3, handletextpad=0.4, columnspacing=1.4)
    figure.tight_layout(pad=0.3, w_pad=2.2, rect=(0.0, 0.0, 1.0, 0.9))
    shared.save_figure(figure, "speedup", figures_dir)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Prose
# ---------------------------------------------------------------------------

def median(values):
    return statistics.median(values)


def add_cusparse_fp16(table, csv_path):
    """Add to the campaign the FP16 cuSPARSE times of a later run, which also timed CSR-i32.

    The campaign asked cuSPARSE for an FP16 compute type, which it rejects, so it has no FP16 vendor
    time. The later run computes FP16 in FP32, as our kernels do, and times CSR-i32 beside it; a
    cuSPARSE time enters the campaign as its ratio to that CSR-i32, applied to the CSR-i32 of the
    campaign, so the two runs never compare absolute times. Returns whether any row was added."""
    later = shared.load_measurements(csv_path)
    added = False
    for (matrix, precision), entries in later.items():
        if precision != "fp16" or "i32" not in entries or (matrix, precision) not in table:
            continue
        campaign = table[(matrix, precision)]
        if "i32" not in campaign:
            continue
        for name in shared.CUSPARSE:
            if name not in entries:
                continue
            row = dict(entries[name])
            scale = float(campaign["i32"]["t_ms"]) / float(entries["i32"]["t_ms"])
            row["t_ms"] = str(float(entries[name]["t_ms"]) * scale)
            # the rate follows the rescaled time, so the figure of absolute performance stays consistent
            if "gflops" in row:
                row["gflops"] = str(float(entries[name]["gflops"]) / scale)
            campaign[name] = row
            added = True
    return added


def setup_tex(device, records, identical, cusparse_deviation, timing, admissible_before, cost, own_gain, separate_fp16):
    reps, warmup, min_sample_ms = timing
    total = len(of_precision(records, "fp64"))
    admissible_after = len(with_layout(of_precision(records, "fp64"), "r16"))
    resident = sum(1 for record in of_precision(records, "fp64") if record["resident"])
    all_identical = all(all(flags) for flags in identical.values())
    identity_text = ("and they are on every matrix at all three precisions" if all_identical
                     else "which is checked on every matrix")
    vendor = {precision: median([record["cusparse_over_i32"] for record in of_precision(records, precision)
                                 if record["cusparse_over_i32"] is not None]) for precision in ("fp64", "fp32", "fp16")
              if any(record["cusparse_over_i32"] is not None for record in of_precision(records, precision))}
    vendor_text = (rf"It takes a median ${vendor['fp64']:.2f}$ of the time of CSR-i32 at FP64, ${vendor['fp32']:.2f}$ at FP32 and ${vendor['fp16']:.2f}$ at FP16, where it reads FP16 values and sums in FP32 as the kernels of Sec.~\ref{{sec:model}} do."
                   if "fp16" in vendor else
                   rf"It takes a median ${vendor['fp64']:.2f}$ of the time of CSR-i32 at FP64 and ${vendor['fp32']:.2f}$ at FP32.")
    if "fp16" in vendor and separate_fp16:
        vendor_text += ("\nAt FP16 it was timed in a later run beside CSR-i32, with the same protocol, and its time enters here "
                        "through its ratio to that CSR-i32.")
    brief_identity = ("with results bit-identical to those of CSR-i32 on every matrix" if all_identical
                      else "with results checked against those of CSR-i32 on every matrix")
    brief_vendor = (rf"${vendor['fp64']:.2f}$, ${vendor['fp32']:.2f}$ and ${vendor['fp16']:.2f}$ of the time of CSR-i32 at FP64, FP32 and FP16"
                    if "fp16" in vendor else
                    rf"${vendor['fp64']:.2f}$ and ${vendor['fp32']:.2f}$ of the time of CSR-i32 at FP64 and FP32")
    return rf"""
\subsection{{Setup}}
\label{{sec:results-setup}}

\begin{{brief}}
All measurements run on {device.description()}{device.toolkit()}, and $B$ is its theoretical bandwidth, ${device.theoretical_bw_gbs:.0f}$~GB/s, while Sec.~\ref{{sec:results-rate}} repeats the measurements on a Quadro RTX 5000.
The four layouts run in one GPU kernel with $T$ threads per row and values and vectors in FP64, FP32 or FP16, and each is compared with CSR-i32 at the $T \in \{{2,4,8,16,32\}}$ fastest for CSR-i32, {brief_identity}.
The vendor reference is the faster of the CSR\_ALG1 and CSR\_ALG2 algorithms of \texttt{{cusparseSpMV}}.
Each time is the median of ${reps}$ samples of at least ${min_sample_ms:g}$~ms after ${warmup}$ warm-up products, without flushing the caches, so the {resident} matrices below the ${device.l2_mib:.0f}$~MiB L2 are served from it, the most favourable case for them, since in an iterative solver other kernels can evict the matrix between two products.
These timings, over every matrix, layout and precision, are called the campaign below.
\end{{brief}}
\begin{{extended}}
All measurements run on {device.description()}{device.toolkit()}, while Sec.~\ref{{sec:results-rate}} repeats the measurements on a Quadro RTX 5000.
$B$ is the theoretical bandwidth of the device, ${device.theoretical_bw_gbs:.0f}$~GB/s.
The four layouts of Sec.~\ref{{sec:model}} are measured with values in FP64, FP32 or FP16 and the vectors $x$ and $y$ in the same precision as the values.
All of them run in the same GPU kernel, one vector of $T$ threads per row, and every layout is timed at each $T \in \{{2,4,8,16,32\}}$.
A layout is compared with CSR-i32 at the $T$ that is fastest for CSR-i32 on the same matrix and precision, so the two share the balance of the work across threads (Sec.~\ref{{sec:model}}).
\extonly{{A compressed layout is sometimes fastest at a different $T$, at FP64 on {own_gain[('fp64', 'r16')][0]} of the {own_gain[('fp64', 'r16')][1]} matrices for CSR-r16 and {own_gain[('fp64', 'lp')][0]} of {own_gain[('fp64', 'lp')][1]} for CSR-lp, where its own $T$ would make it faster by a median ${100.0 * (own_gain[('fp64', 'r16')][2] - 1.0):.1f}$ and ${100.0 * (own_gain[('fp64', 'lp')][2] - 1.0):.1f}$ per cent, a gain of tuning rather than of the encoding.}}
At a given precision the layouts keep the value array and the order of the nonzeros, so at every $T$ their results must be bit-identical to those of CSR-i32, {identity_text}.
Every matrix is reordered with a symmetric Reverse Cuthill--McKee permutation before it is measured, which makes CSR-r16 exact on {admissible_after} of the {total} matrices instead of {admissible_before}.
The permutation is computed once per matrix, and its cost is not part of the times, which are those of the kernel alone.

The vendor reference is the faster of the CSR\_ALG1 and CSR\_ALG2 algorithms of \texttt{{cusparseSpMV}} on the same arrays as CSR-i32.
{vendor_text}
\extonly{{Its result differs from that of CSR-i32 by at most $\max_r |\Delta y_r| / \max_r |y_r| = {shared.scientific(cusparse_deviation['fp64'])}$ in FP64 and ${shared.scientific(cusparse_deviation['fp32'])}$ in FP32, since it sums in a different order.}}
Each time is the median of ${reps}$ samples after ${warmup}$ warm-up products, each sample running enough consecutive products to last at least ${min_sample_ms:g}$~ms.
These timings, over every matrix, layout and precision, are called the campaign below.
The products run back to back without flushing the caches, so the {resident} matrices whose working set at FP64 is below the ${device.l2_mib:.0f}$~MiB L2 are served from it.
This is the most favourable case for these matrices, since the same matrix and the same $x$ stay in the L2 from one product to the next, while in an iterative solver $x$ changes at every iteration and the other kernels stream their vectors through the L2 between two products and can evict the matrix.
On the matrices above the L2 the matrix is read again from the DRAM at every product in either case.
\end{{extended}}
"""


def admissibility_tex(records, characterization):
    reference = of_precision(records, "fp64")
    total = len(reference)
    absolute = len(with_layout(reference, "i16"))
    relative = len(with_layout(reference, "r16"))
    excluded = [record["matrix"] for record in reference if "r16" not in record["layouts"]]
    by_domain = {}
    for matrix in excluded:
        domain = characterization[matrix]["domain"]
        by_domain[domain] = by_domain.get(domain, 0) + 1
    leading = sorted(by_domain.items(), key=lambda item: -item[1])[:2]
    leading_text = " and ".join(rf"{shared.DOMAIN_PROSE[domain]} ({count})" for domain, count in leading)
    bits = median([record["layouts"]["lp"]["bits"] for record in reference])
    return rf"""
\subsection{{Applicability of each layout}}
\label{{sec:results-admissible}}

CSR-i16 is exact on the {absolute} matrices with $n \le 2^{{16}}$, CSR-r16 on {relative} of the {total}, and CSR-lp on all of them.
Of the {len(excluded)} matrices CSR-r16 cannot hold, most are {leading_text}, whose rows still span more than $2^{{16}}$ columns after the permutation.
CSR-lp spends a median of ${bits:.1f}$ bits per nonzero on its packed differences against the $32$ of CSR-i32.
"""


def geometric_mean(values):
    return math.exp(sum(math.log(value) for value in values) / len(values))


def gain_table_lines(records):
    """Geometric means of the two factors and of the speedup, per regime, so that the
    table shows where the byte ratio alone predicts the speedup and where the rate ratio moves it."""
    lines = []
    ceiling_cells = " & ".join(rf"\multicolumn{{3}}{{c}}{{${median([record['r_max'] for record in of_precision(records, precision)]):.2f}$}}"
                               for precision, _, _ in PRECISIONS)
    lines.append(rf"$r_{{\max}}$ & & {ceiling_cells} \\")
    for regime, regime_label, _ in LIMIT_REGIMES:
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{11}}{{l}}{{\emph{{{regime_label}}}}} \\")
        for layout, label, _ in LIMIT_LAYOUTS:
            cells = []
            for precision, _, _ in PRECISIONS:
                group = [record for record in with_layout(of_precision(records, precision), layout)
                         if regime_of(record) == regime]
                if not group:
                    cells += ["--", "--", "--"]
                    continue
                cells.append(rf"${geometric_mean([record['layouts'][layout]['byte_ratio'] for record in group]):.2f}$")
                cells.append(rf"${geometric_mean([record['layouts'][layout]['speedup'] / record['layouts'][layout]['byte_ratio'] for record in group]):.2f}$")
                cells.append(rf"${geometric_mean([record['layouts'][layout]['speedup'] for record in group]):.2f}$")
            count = len([record for record in with_layout(of_precision(records, "fp64"), layout) if regime_of(record) == regime])
            lines.append(rf"\quad {label} & {count} & " + " & ".join(cells) + r" \\")
    return lines


def performance_tex(table, device):
    matrices, summary = performance_summary(table, device)
    cell = lambda precision, layout, index: summary[(precision, layout)][index]
    triple = lambda precision, index: (f"{cell(precision, 'i32', index):.0f}, {cell(precision, 'r16', index):.0f} "
                                       f"and {cell(precision, 'lp', index):.0f}")
    return rf"""
Fig.~\ref{{fig:performance}} states the same measurements in absolute terms on the {len(matrices)} matrices on which CSR-r16 and CSR-lp are both exact and whose working set at FP64 exceeds the L2, each bar against the memory roof $2B/q$ of its own traffic.
A narrower index raises the bar and its roof together, while the distance from each bar to its own roof widens as the values narrow.

\begin{{figure}}[t]
  \centering
  \includegraphics[width=\columnwidth]{{figures/performance.pdf}}
  \caption{{Performance of each layout at the three value precisions on the {len(matrices)} matrices on which CSR-r16 and CSR-lp are both exact and whose working set at FP64 exceeds the L2, median with the interquartile range, and as a tick the memory roof $2B/q$ of that configuration.
  cuSPARSE at FP16 reads FP16 values and sums in FP32.}}
  \label{{fig:performance}}
\end{{figure}}
"""


def gain_tex(records, vendor_profile=None):
    regime_rate = {(regime, layout, precision): geometric_mean([record['layouts'][layout]['speedup'] / record['layouts'][layout]['byte_ratio']
                                                           for record in with_layout(of_precision(records, precision), layout)
                                                           if regime_of(record) == regime])
                   for regime in ('domain', 'uneven', 'cache') for layout in ('r16', 'lp') for precision, _, _ in PRECISIONS
                   if any(regime_of(record) == regime for record in with_layout(of_precision(records, precision), layout))}
    table_block = "\n".join(gain_table_lines(records))
    second = {precision: median([record["layouts"]["r16"]["speedup"] / record["layouts"]["r16"]["byte_ratio"]
                                 for record in with_layout(of_precision(records, precision), "r16")])
              for precision, _, _ in PRECISIONS}
    speedup = {precision: median([record["layouts"]["r16"]["speedup"]
                                  for record in with_layout(of_precision(records, precision), "r16")])
               for precision, _, _ in PRECISIONS}
    ceiling = {precision: median([record["r_max"] for record in of_precision(records, precision)])
               for precision, _, _ in PRECISIONS}
    shared_sets = {precision: with_layout(with_layout(of_precision(records, precision), "r16"), "lp")
                   for precision, _, _ in PRECISIONS}
    lane_fewer_bytes = {precision: sum(1 for record in group
                                       if record["layouts"]["lp"]["byte_ratio"] > record["layouts"]["r16"]["byte_ratio"])
                        for precision, group in shared_sets.items()}
    lane_time_over_relative = {precision: median([record["layouts"]["r16"]["speedup"] / record["layouts"]["lp"]["speedup"]
                                                  for record in group])
                               for precision, group in shared_sets.items()}
    absolute_fp16 = with_layout(of_precision(records, "fp16"), "i16")
    absolute_fp64 = with_layout(of_precision(records, "fp64"), "i16")
    absolute_resident = sum(1 for record in absolute_fp64 if record["resident"])
    absolute_tiny = {precision: sum(1 for record in with_layout(of_precision(records, precision), "i16") if record["working_set"] < 0.5)
                     for precision, _, _ in PRECISIONS}
    absolute_time = {precision: median([1000.0 * record["time_ms"] for record in with_layout(of_precision(records, precision), "i16")])
                     for precision, _, _ in PRECISIONS}
    absolute_outside = sorted(record["layouts"]["i16"]["speedup"] for record in absolute_fp16 if not record["resident"])
    absolute_outside_text = " and ".join(rf"${value:.2f}\times$" for value in absolute_outside)
    vendor_wins = {precision: sum(1 for record in with_layout(of_precision(records, precision), "r16")
                                  if (record["layouts"]["r16"]["over_cusparse"] or 0.0) > 1.0)
                   for precision in ("fp64", "fp32", "fp16")}
    relative_count = len(with_layout(of_precision(records, "fp64"), "r16"))
    vendor_even = {}
    for precision, _, _ in PRECISIONS:
        ratios = [record["layouts"]["r16"]["over_cusparse"] for record in with_layout(of_precision(records, precision), "r16")
                  if regime_of(record) == "domain" and record["layouts"]["r16"]["over_cusparse"]]
        vendor_even[precision] = (geometric_mean(ratios), sum(1 for value in ratios if value < 1.0), len(ratios))
    vendor_text = (rf"On balanced matrices CSR-r16 is also faster than cuSPARSE, by a geometric mean of ${vendor_even['fp64'][0]:.2f}\times$, ${vendor_even['fp32'][0]:.2f}\times$ and ${vendor_even['fp16'][0]:.2f}\times$ at FP64, FP32 and FP16, "
                   rf"and slower on {vendor_even['fp64'][1]}, {vendor_even['fp32'][1]} and {vendor_even['fp16'][1]} of {vendor_even['fp64'][2]}, {vendor_even['fp32'][2]} and {vendor_even['fp16'][2]} matrices.")
    if vendor_profile:
        vendor_text += ("\n" + rf"\extonly{{Nsight Compute on {vendor_profile['matrices']} of the matrices shows that each call of cuSPARSE also runs, before its SpMV kernel, a kernel of ${vendor_profile['scale'][0]:.1f}$ to ${vendor_profile['scale'][1]:.1f}$~\textmu s that scales $y$, which weighs only on short products.}}")
    relative_wins = {precision: sum(1 for record in with_layout(of_precision(records, precision), "r16")
                                    if record["layouts"]["r16"]["speedup"] > 1.0) for precision, _, _ in PRECISIONS}
    captured = {precision: 100.0 * median([(record["layouts"]["r16"]["byte_ratio"] - 1.0) / (record["r_max"] - 1.0)
                                         for record in with_layout(of_precision(records, precision), "r16")])
                for precision, _, _ in PRECISIONS}
    losses = {}
    for precision, _, _ in PRECISIONS:
        lane = with_layout(of_precision(records, precision), "lp")
        slower = [record for record in lane if record["layouts"]["lp"]["speedup"] < 1.0]
        losses[precision] = (len(slower), len(lane),
                             median([record["layouts"]["lp"]["speedup"] for record in slower]) if slower else 1.0,
                             min([record["layouts"]["lp"]["speedup"] for record in slower]) if slower else 1.0,
                             sum(1 for record in slower if record["layouts"]["lp"]["byte_ratio"] <= 1.0),
                             median([record["row_length"] for record in slower
                                     if record["layouts"]["lp"]["byte_ratio"] <= 1.0] or [0.0]))
    slow = {key: [] for key in ("bytes", "uneven", "cache", "even")}
    slow_even = []
    lowest = (2.0, "")
    for precision, _, _ in PRECISIONS:
        counts = {key: 0 for key in slow}
        for record in with_layout(of_precision(records, precision), "lp"):
            measured = record["layouts"]["lp"]
            if measured["speedup"] >= 1.0:
                continue
            if measured["speedup"] < lowest[0]:
                lowest = (measured["speedup"], record["matrix"])
            if measured["byte_ratio"] <= 1.0:
                counts["bytes"] += 1
            elif record["resident"]:
                counts["cache"] += 1
            elif record["imbalance"] > IMBALANCE_LIMIT:
                counts["uneven"] += 1
            else:
                counts["even"] += 1
                slow_even.append(measured)
        for key in slow:
            slow[key].append(counts[key])
    lowest_lane = shared.tex_escape(lowest[1])
    slow_even_ratio = max([measured["byte_ratio"] for measured in slow_even] or [1.0])
    slow_even_speedup = min([measured["speedup"] for measured in slow_even] or [1.0])
    shared_count = len(shared_sets["fp64"])
    fewer_share = [lane_fewer_bytes[precision] / len(group) for precision, group in shared_sets.items()]
    fewer_text = ("about two thirds of them at every precision" if all(0.6 <= share <= 0.73 for share in fewer_share)
                  else f"{lane_fewer_bytes['fp64']}, {lane_fewer_bytes['fp32']} and {lane_fewer_bytes['fp16']} of them at FP64, FP32 and FP16")
    lane_versus_relative = (
        rf"\extonly{{On the {shared_count} matrices on which both apply, CSR-lp moves fewer bytes than CSR-r16 on {fewer_text}, "
        rf"yet its time over that of CSR-r16 has a median of ${lane_time_over_relative['fp64']:.2f}$ at FP64, "
        rf"${lane_time_over_relative['fp32']:.2f}$ at FP32 and ${lane_time_over_relative['fp16']:.2f}$ at FP16, "
        rf"so on these matrices what it gains in bytes it loses in $K$, and more so as the values narrow.}}")
    byte_ratio = {precision: median([record["layouts"]["r16"]["byte_ratio"]
                                     for record in with_layout(of_precision(records, precision), "r16")])
                  for precision, _, _ in PRECISIONS}
    return rf"""
\subsection{{Speedup of the compressed layouts}}
\label{{sec:results-gain}}

Table~\ref{{tab:results-gain}} gives the geometric means of the byte ratio $q_{{\mathrm{{ref}}}}/q$ of Eq.~\eqref{{eq:q-layouts}}, of the rate ratio $K/K_{{\mathrm{{ref}}}}$ and of the speedup over CSR-i32 in three regimes, on balanced matrices, with a working set above the L2 and the longest row at most {IMBALANCE_LIMIT:g} times the mean, on unbalanced matrices, and in the L2, and Fig.~\ref{{fig:speedup}} gives the speedup of each layout on every matrix, sorted.
On balanced matrices the rate ratio is close to one at FP64 and falls as the values narrow, while on unbalanced matrices and in the L2 it departs further from one, for the reasons of Sec.~\ref{{sec:results-limits}}.
No byte ratio exceeds $r_{{\max}}$, the first row of the table, and since CSR-r16 keeps two of the four index bytes, it obtains a median ${captured['fp64']:.0f}$, ${captured['fp32']:.0f}$ and ${captured['fp16']:.0f}$ per cent of the byte saving that $r_{{\max}}$ allows at FP64, FP32 and FP16, so the room left to an index narrower than 16 bits grows as the values narrow.
{vendor_text}
\extonly{{Unlike the first factor, the rate ratio is known only once both kernels have been timed, since $K$ is $Q$ over the measured time, and Sec.~\ref{{sec:results-rate}} measures what it consists of.}}
\begin{{brief}}
CSR-i16 is exact only on {len(absolute_fp64)} small matrices, of which {absolute_tiny['fp64']} at FP64 and {absolute_tiny['fp16']} at FP16 lie below half of the L2, where the time follows launch and latency rather than bytes, so its median speedup drops from ${median([record["layouts"]["i16"]["speedup"] for record in absolute_fp64]):.2f}\times$ at FP64 to ${median([record["layouts"]["i16"]["speedup"] for record in absolute_fp16]):.2f}\times$ at FP16.
CSR-lp is slower than CSR-i32 on {losses['fp64'][0]}, {losses['fp32'][0]} and {losses['fp16'][0]} of its {losses['fp64'][1]} matrices at FP64, FP32 and FP16, almost all unbalanced, in the L2, or with rows so short that it moves at least as many bytes as CSR-i32.
\end{{brief}}
\begin{{extended}}
From FP64 to FP16 the median byte ratio of CSR-i16 grows from ${median([record["layouts"]["i16"]["byte_ratio"] for record in absolute_fp64]):.2f}$ to ${median([record["layouts"]["i16"]["byte_ratio"] for record in absolute_fp16]):.2f}$, while its median speedup drops from ${median([record["layouts"]["i16"]["speedup"] for record in absolute_fp64]):.2f}\times$ to ${median([record["layouts"]["i16"]["speedup"] for record in absolute_fp16]):.2f}\times$.
CSR-i16 is exact only with at most $2^{{16}}$ columns, so its {len(absolute_fp64)} matrices are small, and narrower values make them smaller still, so that {absolute_tiny['fp64']} of them at FP64 and {absolute_tiny['fp16']} at FP16 lie below half of the L2, where CSR-i32 takes a median of ${absolute_time['fp64']:.0f}$ and ${absolute_time['fp16']:.0f}$~\textmu s at FP64 and FP16 and the time follows launch and latency rather than bytes (Sec.~\ref{{sec:results-limits}}).
On the {len(absolute_outside)} of them that stay outside the L2 at FP16 its speedup is {absolute_outside_text}.
CSR-lp is slower than CSR-i32 on {losses['fp64'][0]}, {losses['fp32'][0]} and {losses['fp16'][0]} of its {losses['fp64'][1]} matrices at FP64, FP32 and FP16, and almost all of them fail one of the conditions of Sec.~\ref{{sec:results-limits}}.
On {slow['bytes'][0]}, {slow['bytes'][1]} and {slow['bytes'][2]} of them it moves at least as many bytes as CSR-i32, since their rows hold a median of ${losses['fp64'][5]:.1f}$ nonzeros and the nine bytes per row of Eq.~\eqref{{eq:q-lane}} outweigh the index bits saved.
On {slow['uneven'][0]}, {slow['uneven'][1]} and {slow['uneven'][2]} the longest rows set the time and the decoding lengthens them, down to ${min(losses[p][3] for p, _, _ in PRECISIONS):.2f}\times$ on \texttt{{{lowest_lane}}}, and on {slow['cache'][0]}, {slow['cache'][1]} and {slow['cache'][2]} the matrix fits in the L2, more of them as the values narrow and the working set shrinks.
The remaining {slow['even'][0]}, {slow['even'][1]} and {slow['even'][2]}, balanced and outside the L2, have a byte ratio of at most ${slow_even_ratio:.2f}$ and lose at most ${100.0 * (1.0 - slow_even_speedup):.0f}$ per cent.
\end{{extended}}
{lane_versus_relative}

\begin{{table}}[t]
\centering
\caption{{\scriptsize Geometric means of the byte ratio, the rate ratio and the speedup over CSR-i32 per regime, after RCM. The first row is the bound $r_{{\max}}$ of Eq.~\eqref{{eq:rmax}}, over all {len(of_precision(records, 'fp64'))} matrices. The count $n$ is that at FP64, and as the values narrow some matrices move into the L2.}}
\label{{tab:results-gain}}
\footnotesize
\setlength{{\tabcolsep}}{{2.2pt}}
\begin{{tabular}}{{lcrrrrrrrrr}}
\toprule
 & & \multicolumn{{3}}{{c}}{{FP64}} & \multicolumn{{3}}{{c}}{{FP32}} & \multicolumn{{3}}{{c}}{{FP16}} \\
\cmidrule(lr){{3-5}}\cmidrule(lr){{6-8}}\cmidrule(lr){{9-11}}
layout & $n$ & $\frac{{q_{{\mathrm{{ref}}}}}}{{q}}$ & $\frac{{K}}{{K_{{\mathrm{{ref}}}}}}$ & $S$ & $\frac{{q_{{\mathrm{{ref}}}}}}{{q}}$ & $\frac{{K}}{{K_{{\mathrm{{ref}}}}}}$ & $S$ & $\frac{{q_{{\mathrm{{ref}}}}}}{{q}}$ & $\frac{{K}}{{K_{{\mathrm{{ref}}}}}}$ & $S$ \\
\midrule
{table_block}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure*}}[t]
  \centering
  \includegraphics[width=\textwidth]{{figures/speedup.pdf}}
  \caption{{Speedup over CSR-i32 of each layout on the RCM-ordered matrices on which it is exact, sorted from the lowest to the highest, with the number of those matrices in the legend.
  The tick and the count where a curve crosses one give the matrices on which the layout is slower than CSR-i32.}}
  \label{{fig:speedup}}
\end{{figure*}}
"""


def band_tex(records):
    counts, width, uneven_below = {}, {}, {}
    for precision, _, _ in PRECISIONS:
        for layout, _ in BAND_LAYOUTS:
            group = band_subset(records, precision, layout)
            counts[(precision, layout)] = (sum(1 for record in group if inside_band(record["layouts"][layout])),
                                           len(group))
        width[precision] = median([record["layouts"]["r16"]["band"][1] / record["layouts"]["r16"]["band"][0]
                                   for record in band_subset(records, precision, "r16")])
        uneven = [record for record in with_layout(of_precision(records, precision), "lp")
                  if not record["resident"] and record["imbalance"] > IMBALANCE_LIMIT]
        uneven_below[precision] = (sum(1 for record in uneven
                                       if record["layouts"]["lp"]["speedup"] < record["layouts"]["lp"]["band"][0]),
                                   len(uneven))
    served = sorted(record["imbalance"] for record in of_precision(records, "fp64") if not record["resident"])
    below_limit = max(value for value in served if value <= IMBALANCE_LIMIT)
    above_limit = min(value for value in served if value > IMBALANCE_LIMIT)
    narrow = [record for record in band_subset(records, "fp32", "r16")
              if record["layouts"]["r16"]["band"][1] / record["layouts"]["r16"]["band"][0] < 1.01]
    lane_above = [record["layouts"]["lp"]["speedup"] / record["layouts"]["lp"]["byte_ratio"]
                  for record in band_subset(records, "fp64", "lp")
                  if record["layouts"]["lp"]["speedup"] > record["layouts"]["lp"]["band"][1]]
    cell = lambda precision, layout: f"{counts[(precision, layout)][0]} of {counts[(precision, layout)][1]}"
    outside = [max(record["layouts"]["r16"]["speedup"] / record["layouts"]["r16"]["band"][1],
                   record["layouts"]["r16"]["band"][0] / record["layouts"]["r16"]["speedup"]) - 1.0
               for precision, _, _ in PRECISIONS for record in band_subset(records, precision, "r16")
               if not inside_band(record["layouts"]["r16"])]
    narrow_text = (rf"The speedups outside it lie at most ${100.0 * max(outside or [0.0]):.1f}$ per cent from an end, and the one farthest in units of the band is \texttt{{{narrow[0]['matrix'].replace('_', chr(92) + '_')}}}, whose rows hold ${narrow[0]['row_length']:.1f}$ nonzeros on average, so that CSR-r16 saves almost nothing there and its band is narrower than one per cent." + "\n"
                   if narrow else "")
    return rf"""
\begin{{extended}}
\subsection{{The gather band}}
\label{{sec:results-band}}

The band of Eq.~\eqref{{eq:band}} assumes that the two runs retire their real bytes at the same rate, so it is tested where nothing but the gather should separate them, on the matrices served from the DRAM whose longest row is at most {IMBALANCE_LIMIT:g} times the mean.
That limit falls in the gap between ${below_limit:.1f}$ and ${above_limit:.1f}$ of the distribution of that ratio over the matrices served from the DRAM.
Fig.~\ref{{fig:band}} draws the band of every such matrix on the axis of the speedup together with the measured value.
CSR-r16 falls inside it on {cell('fp64', 'r16')} matrices at FP64, {cell('fp32', 'r16')} at FP32 and {cell('fp16', 'r16')} at FP16.
The band contains these speedups through its width rather than through the gather, since on balanced matrices the DRAM delivers the counted bytes within a few per cent and the shortfall from the byte ratio is the lower bandwidth of the compressed kernel (Sec.~\ref{{sec:results-rate}}).
{narrow_text}The band is a median ${width['fp64']:.2f}\times$ wide at FP64, ${width['fp32']:.2f}\times$ at FP32 and ${width['fp16']:.2f}\times$ at FP16.
Its lower end charges a whole $32$-byte sector to every gather, which is the most a gather can cost, and a sector weighs more against narrower values, so the band constrains the speedup at FP64 far more than at FP16.
CSR-lp falls inside it on {cell('fp64', 'lp')}, {cell('fp32', 'lp')} and {cell('fp16', 'lp')} matrices.
At FP64 it lies above the band on {len(lane_above)} of them, with $K/K_{{\mathrm{{ref}}}}$ at most ${max(lane_above) if lane_above else 1.0:.2f}$, which the band, assuming equal rates, does not allow for.
On the matrices served from the DRAM with longer rows than the limit, it falls below the band on {uneven_below['fp64'][0]} of {uneven_below['fp64'][1]} at FP64, {uneven_below['fp32'][0]} of {uneven_below['fp32'][1]} at FP32 and {uneven_below['fp16'][0]} of {uneven_below['fp16'][1]} at FP16, where the time follows the longest rows rather than the traffic (Sec.~\ref{{sec:results-rate}}).

\begin{{figure*}}[t]
  \centering
  \includegraphics[width=\textwidth]{{figures/band.pdf}}
  \caption{{Gather band of Eq.~\eqref{{eq:band}} (grey bars) and measured speedup over CSR-i32 (dots, crosses when outside the band) on the matrices served from the DRAM whose longest row is at most {IMBALANCE_LIMIT:g} times the mean, sorted by the upper end of the band, $q_{{\mathrm{{ref}}}}/q$.
  The lower end of each bar charges a $32$-byte sector to every gather on $x$, the upper end reads $x$ once.}}
  \label{{fig:band}}
\end{{figure*}}
\end{{extended}}
"""


RATE_GROUPS = (("domain", "balanced"), ("gather", "short rows"), ("uneven", "unbalanced"), ("cache", "in the L2"))
RATE_GROUP_OF_COMMENT = {"# inside the domain": "domain", "# working set inside the L2": "cache",
                         "# imbalance above ten": "uneven", "# short rows": "gather"}


def rate_groups(list_path):
    """Matrix -> group of experiments/lists/k_study.txt, read off the comment above each block."""
    groups, current = {}, None
    for line in list_path.read_text().splitlines():
        for prefix, group in RATE_GROUP_OF_COMMENT.items():
            if line.startswith(prefix):
                current = group
        if line.strip() and not line.startswith("#"):
            groups[Path(line.strip()).stem] = current
    return groups


def rate_measurements(study_dir, list_path, original_csv):
    """Per profiled matrix, precision and compressed layout, at the threads per row fastest for CSR-i32
    in the campaign of the original ordering: the rate ratio K/K_ref from the profiled times and the
    counted bytes, the counted over the DRAM bytes of CSR-i32 and of the layout, the busiest SM over the
    mean for CSR-i32, and the DRAM bytes per global load instruction of the layout over CSR-i32."""
    from analyse_stalls import read_ncu
    groups = rate_groups(list_path)
    counted, nonzeros, reference_threads = {}, {}, {}
    names = {"lane-packed": "lp", "row-base": "r16"}
    for row in shared.read_rows(original_csv):
        if row["status"] != "ok" or row["row_encoding"] != "cumulative" or row["kernel"] != "custom":
            continue
        layout = names.get(row["column_mode"]) or ("i32" if row["column_mode"] == "absolute"
                                                    and row["column_bytes"] == "4" else None)
        if layout is None:
            continue
        counted[(row["matrix"], row["prec"], layout)] = float(row["bytes_per_nnz"]) * int(row["nnz"])
        nonzeros[row["matrix"]] = int(row["nnz"])
        if layout == "i32":
            reference_threads[(row["matrix"], row["prec"])] = int(row["best_tpr"])
    measurements = []
    for path in sorted(study_dir.glob("*.ncu.csv")):
        matrix = path.name.split(".")[0]
        kernels = read_ncu(path)
        for precision, _, _ in PRECISIONS:
            threads = reference_threads.get((matrix, precision))
            reference = kernels.get(("i32", precision, threads))
            if reference is None:
                continue
            for layout in ("r16", "lp"):
                candidate = kernels.get((layout, precision, threads))
                if candidate is None or (matrix, precision, layout) not in counted:
                    continue
                rate = lambda counters, name: counted[(matrix, precision, name)] / counters["gpu__time_duration.sum"]
                per_load = lambda counters: (counters["dram__bytes_read.sum"]
                                             / counters["smsp__inst_executed_op_global_ld.sum"])
                measurements.append({
                    "matrix": matrix, "group": groups.get(matrix), "precision": precision, "layout": layout,
                    "rate_ratio": rate(candidate, layout) / rate(reference, "i32"),
                    "counted_reference": counted[(matrix, precision, "i32")] / reference["dram__bytes_read.sum"],
                    "counted_layout": counted[(matrix, precision, layout)] / candidate["dram__bytes_read.sum"],
                    "spread": reference["sm__cycles_active.max"] / reference["sm__cycles_active.avg"],
                    "per_load": per_load(candidate) / per_load(reference),
                    "instructions": candidate["smsp__inst_executed.sum"] / reference["smsp__inst_executed.sum"],
                    "busy_reference": reference["dram__throughput.avg.pct_of_peak_sustained_elapsed"],
                    "busy_layout": candidate["dram__throughput.avg.pct_of_peak_sustained_elapsed"],
                    "occupancy_reference": reference["sm__warps_active.avg.pct_of_peak_sustained_active"],
                    "occupancy_layout": candidate["sm__warps_active.avg.pct_of_peak_sustained_active"],
                    "memory_reference": reference["smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"],
                    "memory_layout": candidate["smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"],
                    "wait_reference": reference["smsp__warp_issue_stalled_wait_per_warp_active.pct"],
                    "wait_layout": candidate["smsp__warp_issue_stalled_wait_per_warp_active.pct"]})
    return measurements


def regime_of(record):
    """Where a matrix lies against the domain of the model: in the L2, with uneven rows, or inside."""
    if record["resident"]:
        return "cache"
    return "uneven" if record["imbalance"] > IMBALANCE_LIMIT else "domain"


def validity(records):
    """Per regime, compressed layout and precision: the number of matrices, the median of K/K_ref, and
    the matrices on which the byte ratio predicts the speedup within PREDICTION_TOLERANCE."""
    result = {}
    for precision, _, _ in PRECISIONS:
        for layout in ("r16", "lp"):
            for regime in ("domain", "uneven", "cache"):
                ratios = [record["layouts"][layout]["speedup"] / record["layouts"][layout]["byte_ratio"]
                          for record in with_layout(of_precision(records, precision), layout)
                          if regime_of(record) == regime]
                if ratios:
                    result[(regime, layout, precision)] = (
                        len(ratios), median(ratios),
                        sum(1 for ratio in ratios if abs(ratio - 1.0) <= PREDICTION_TOLERANCE),
                        min(ratios), max(ratios))
    return result


LIMIT_REGIMES = (("domain", "balanced", shared.SERIES_BLUE), ("uneven", "unbalanced", shared.SERIES_ORANGE),
                 ("cache", "in the L2", shared.SERIES_AQUA))
LIMIT_LAYOUTS = (("r16", "CSR-r16", "s"), ("lp", "CSR-lp", "^"))


def limits_points(records):
    """Every admissible compressed layout of every matrix and precision, with its regime, q/q_ref and K/K_ref."""
    return [{"precision": record["precision"], "layout": layout, "regime": regime_of(record),
             "working_set": record["working_set"], "imbalance": record["imbalance"], "resident": record["resident"],
             "matrix": record["matrix"], "time_us": 1000.0 * record["time_ms"],
             "speedup": measured["speedup"], "byte_ratio": measured["byte_ratio"], "band": measured["band"],
             "load_ratio": 1.0 / measured["byte_ratio"], "rate_ratio": measured["speedup"] / measured["byte_ratio"]}
            for record in records for layout, measured in record["layouts"].items() if layout in ("r16", "lp")]


def limits_figure(points, measurements, records, figures_dir):
    """Each factor against its indicator: in the top row K/K_ref over the campaign, each panel on the
    points that satisfy the other two conditions, in the bottom row the counter of Nsight Compute that
    explains the factor, on the profiled matrices, against the same indicator."""
    panels = (
        ("working_set", r"working set / L2", lambda point: point["imbalance"] <= IMBALANCE_LIMIT, 1.0, True),
        ("imbalance", r"longest / mean row, $\ell_{\max}/\ell$", lambda point: not point["resident"], IMBALANCE_LIMIT, True),
        ("load_ratio", r"bytes per load, $q/q_{\mathrm{ref}}$",
         lambda point: not point["resident"] and point["imbalance"] <= IMBALANCE_LIMIT, None, False),
    )
    figure, axes = plt.subplots(2, 3, figsize=(7.16, 3.2), sharex="col")
    for axis, (key, label, keep, threshold, logarithmic) in zip(axes[0], panels):
        axis.axhspan(1.0 - PREDICTION_TOLERANCE, 1.0 + PREDICTION_TOLERANCE, color="0.92", zorder=0)
        axis.axhline(1.0, color=shared.INK_PRIMARY, linewidth=0.6, zorder=1)
        if threshold is not None:
            axis.axvline(threshold, color=shared.INK_SECONDARY, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
        for precision, _, _ in reversed(PRECISIONS):
            for layout, _, marker in LIMIT_LAYOUTS:
                chosen = [point for point in points if point["precision"] == precision and point["layout"] == layout and keep(point)]
                axis.scatter([point[key] for point in chosen], [point["rate_ratio"] for point in chosen], s=9, marker=marker,
                             color=shared.PRECISION_COLOUR[precision], alpha=0.75, linewidths=0, zorder=2)
        if key == "load_ratio":
            diagonal = [0.55, 1.0]
            axis.plot(diagonal, diagonal, color=shared.INK_SECONDARY, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
            axis.annotate("$S = 1$", xy=(0.66, 0.5), color=shared.INK_SECONDARY, fontsize=6)
        if logarithmic:
            axis.set_xscale("log")
        axis.set_ylim(0.2, 1.45)
        shared.style_axis(axis)
    for axis in axes[0][1:]:
        axis.set_yticklabels([])
    axes[0][0].set_ylabel(r"$K/K_{\mathrm{ref}}$, campaign")
    axes[0][0].set_title("A. cache, balanced", fontsize=7, loc="left")
    axes[0][1].set_title("B. balance, outside the L2", fontsize=7, loc="left")
    axes[0][2].set_title("C. bytes in flight, balanced", fontsize=7, loc="left")

    facts = {(record["matrix"], record["precision"]): record for record in records}
    profiled = [(entry, facts[(entry["matrix"], entry["precision"])]) for entry in measurements
                if (entry["matrix"], entry["precision"]) in facts]
    cache, balance, flight = axes[1]
    for precision, _, _ in reversed(PRECISIONS):
        colour = shared.PRECISION_COLOUR[precision]
        for layout, _, marker in LIMIT_LAYOUTS:
            chosen = [(entry, record) for entry, record in profiled if entry["precision"] == precision
                      and entry["layout"] == layout]
            even = [(entry, record) for entry, record in chosen if record["imbalance"] <= IMBALANCE_LIMIT]
            cache.scatter([record["working_set"] for _, record in even], [1.0 / entry["counted_layout"] for entry, _ in even],
                          s=12, marker=marker, color=colour, alpha=0.8, linewidths=0, zorder=2)
            inside = [(entry, record) for entry, record in even if not record["resident"]]
            flight.scatter([entry["per_load"] for entry, _ in inside],
                           [entry["busy_layout"] / entry["busy_reference"] for entry, _ in inside],
                           s=12, marker=marker, color=colour, alpha=0.8, linewidths=0, zorder=2)
        spread = {(entry["matrix"], entry["precision"]): (record["imbalance"], entry["spread"]) for entry, record in profiled
                  if entry["precision"] == precision and not record["resident"]}
        balance.scatter([value[0] for value in spread.values()], [value[1] for value in spread.values()], s=12, marker="o",
                        color=colour, alpha=0.8, linewidths=0, zorder=2)
    cache.axhline(1.0, color=shared.INK_PRIMARY, linewidth=0.6, zorder=1)
    cache.axvline(1.0, color=shared.INK_SECONDARY, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
    cache.set_ylabel(r"DRAM bytes / $Q$", labelpad=1)
    balance.axhline(1.0, color=shared.INK_PRIMARY, linewidth=0.6, zorder=1)
    balance.axvline(IMBALANCE_LIMIT, color=shared.INK_SECONDARY, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
    balance.set_ylabel("busiest SM / mean, CSR-i32", labelpad=1)
    flight.axhline(1.0, color=shared.INK_PRIMARY, linewidth=0.6, zorder=1)
    flight.plot([0.5, 1.0], [0.5, 1.0], color=shared.INK_SECONDARY, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
    flight.set_ylabel("DRAM busy / CSR-i32", labelpad=1)
    for axis, label in zip(axes[1], (r"working set / L2", r"longest / mean row, $\ell_{\max}/\ell$",
                                     r"bytes per load / CSR-i32")):
        axis.set_xlabel(label)
        shared.style_axis(axis)
    for axis in axes[1]:
        axis.set_title("Nsight Compute", fontsize=6.5, loc="left", color=shared.INK_SECONDARY)
    handles = [Line2D([], [], marker="o", linestyle="", color=shared.PRECISION_COLOUR[precision], label=label)
               for precision, label, _ in PRECISIONS]
    handles += [Line2D([], [], marker=marker, linestyle="", color="0.4", label=label) for _, label, marker in LIMIT_LAYOUTS]
    figure.legend(handles=handles, loc="upper center", ncol=5, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 1.01))
    figure.tight_layout(rect=(0, 0, 1, 0.95), h_pad=0.6)
    figure.savefig(figures_dir / "limits.pdf")
    plt.close(figure)


def limits_row_figure(points, measurements, records, figures_dir):
    """The three factors in one row for the brief build: the cache as K/K_ref over the campaign, the
    balance and the bytes in flight as the counters of Nsight Compute that explain them."""
    figure, (cache, balance, flight) = plt.subplots(1, 3, figsize=(7.16, 1.85))
    facts = {(record["matrix"], record["precision"]): record for record in records}
    profiled = [(entry, facts[(entry["matrix"], entry["precision"])]) for entry in measurements
                if (entry["matrix"], entry["precision"]) in facts]
    cache.axhspan(1.0 - PREDICTION_TOLERANCE, 1.0 + PREDICTION_TOLERANCE, color="0.92", zorder=0)
    cache.axhline(1.0, color=shared.INK_PRIMARY, linewidth=0.6, zorder=1)
    cache.axvline(1.0, color=shared.INK_SECONDARY, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
    for precision, _, _ in reversed(PRECISIONS):
        colour = shared.PRECISION_COLOUR[precision]
        for layout, _, marker in LIMIT_LAYOUTS:
            chosen = [point for point in points if point["precision"] == precision and point["layout"] == layout
                      and point["imbalance"] <= IMBALANCE_LIMIT]
            cache.scatter([point["working_set"] for point in chosen], [point["rate_ratio"] for point in chosen], s=9,
                          marker=marker, color=colour, alpha=0.75, linewidths=0, zorder=2)
            inside = [(entry, record) for entry, record in profiled if entry["precision"] == precision
                      and entry["layout"] == layout and record["imbalance"] <= IMBALANCE_LIMIT and not record["resident"]]
            flight.scatter([entry["per_load"] for entry, _ in inside],
                           [entry["busy_layout"] / entry["busy_reference"] for entry, _ in inside],
                           s=12, marker=marker, color=colour, alpha=0.8, linewidths=0, zorder=2)
        spread = {(entry["matrix"], entry["precision"]): (record["imbalance"], entry["spread"]) for entry, record in profiled
                  if entry["precision"] == precision and not record["resident"]}
        balance.scatter([value[0] for value in spread.values()], [value[1] for value in spread.values()], s=12, marker="o",
                        color=colour, alpha=0.8, linewidths=0, zorder=2)
    cache.set_xscale("log")
    cache.set_ylim(0.4, 1.45)
    cache.set_xlabel("working set / L2")
    cache.set_ylabel(r"$K/K_{\mathrm{ref}}$", labelpad=1)
    cache.set_title("A. cache, balanced, campaign", fontsize=7, loc="left")
    balance.axhline(1.0, color=shared.INK_PRIMARY, linewidth=0.6, zorder=1)
    balance.axvline(IMBALANCE_LIMIT, color=shared.INK_SECONDARY, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
    balance.set_xscale("log")
    balance.set_xlabel(r"longest / mean row, $\ell_{\max}/\ell$")
    balance.set_ylabel("busiest SM / mean", labelpad=1)
    balance.set_title("B. balance, Nsight Compute", fontsize=7, loc="left")
    flight.axhline(1.0, color=shared.INK_PRIMARY, linewidth=0.6, zorder=1)
    flight.plot([0.5, 1.0], [0.5, 1.0], color=shared.INK_SECONDARY, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
    flight.set_xlabel("bytes per load / CSR-i32")
    flight.set_ylabel("DRAM busy / CSR-i32", labelpad=1)
    flight.set_title("C. bytes in flight, Nsight Compute", fontsize=7, loc="left")
    for axis in (cache, balance, flight):
        shared.style_axis(axis)
    handles = [Line2D([], [], marker="o", linestyle="", color=shared.PRECISION_COLOUR[precision], label=label)
               for precision, label, _ in PRECISIONS]
    handles += [Line2D([], [], marker=marker, linestyle="", color="0.4", label=label) for _, label, marker in LIMIT_LAYOUTS]
    figure.legend(handles=handles, loc="upper center", ncol=5, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 1.04))
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(figures_dir / "limits_brief.pdf")
    plt.close(figure)


def prediction_figure(points, figures_dir):
    """The gather band of every matrix as a grey bar on the speedup axis and the measured speedup as a
    marker, coloured when inside the band and black when outside, matrices sorted by byte ratio, one row
    per regime and one column per precision, each panel on its own speedup axis."""
    in_band = lambda point: point["speedup"] >= point["band"][0] - 1e-9
    figure, axes = plt.subplots(len(LIMIT_REGIMES), len(PRECISIONS), figsize=(7.16, 5.4))
    for row, (regime, regime_label, _) in enumerate(LIMIT_REGIMES):
        for column, (precision, precision_label, _) in enumerate(PRECISIONS):
            axis = axes[row][column]
            colour = shared.PRECISION_COLOUR[precision]
            group = sorted((point for point in points if point["regime"] == regime and point["precision"] == precision),
                           key=lambda point: (point["byte_ratio"], point["layout"]))
            for index, point in enumerate(group):
                axis.plot([index, index], list(point["band"]), color=shared.NEUTRAL_FILL, linewidth=2.4,
                          solid_capstyle="butt", zorder=1)
            for layout, _, marker in LIMIT_LAYOUTS:
                for inside in (True, False):
                    chosen = [(index, point) for index, point in enumerate(group)
                              if point["layout"] == layout and in_band(point) == inside]
                    axis.scatter([index for index, _ in chosen], [point["speedup"] for _, point in chosen], s=10,
                                 marker=marker, color=colour if inside else shared.INK_PRIMARY, linewidths=0, zorder=3)
            axis.axhline(1.0, color=shared.INK_SECONDARY, linewidth=0.6, linestyle=(0, (4, 2)), zorder=0)
            inside_count = sum(1 for point in group if in_band(point))
            axis.set_title(f"{precision_label}, {regime_label}: {len(group) - inside_count}/{len(group)} below", fontsize=7)
            axis.set_xticks([])
            shared.style_axis(axis)
            if column == 0:
                axis.set_ylabel("speedup")
            if row == len(LIMIT_REGIMES) - 1:
                axis.set_xlabel(r"matrices sorted by $q_{\mathrm{ref}}/q$", fontsize=7)
    handles = [Line2D([], [], marker=marker, linestyle="", color="0.4", label=label) for _, label, marker in LIMIT_LAYOUTS]
    handles += [Line2D([], [], color=shared.NEUTRAL_FILL, linewidth=3, label="gather band"),
                Line2D([], [], marker="o", linestyle="", color=shared.INK_PRIMARY, label="below the band")]
    figure.legend(handles=handles, loc="upper center", ncol=4, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 1.01))
    figure.tight_layout(pad=0.3, rect=(0, 0, 1, 0.965))
    figure.savefig(figures_dir / "prediction.pdf")
    plt.close(figure)
    return [point for point in points if point["regime"] == "domain"]


def limits_tex(records, figures_dir, measurements):
    from analyse_stalls import spearman
    points = limits_points(records)
    limits_figure(points, measurements, records, figures_dir)
    limits_row_figure(points, measurements, records, figures_dir)
    predicted = prediction_figure(points, figures_dir)
    below = {regime: (sum(1 for point in points if point['regime'] == regime and point['speedup'] < point['band'][0] - 1e-9),
                      sum(1 for point in points if point['regime'] == regime)) for regime, _, _ in LIMIT_REGIMES}
    regime_band = {regime: (sum(1 for point in points if point['regime'] == regime and point['band'][0] - 1e-9 <= point['speedup'] <= point['band'][1] + 1e-9),
                            sum(1 for point in points if point['regime'] == regime)) for regime, _, _ in LIMIT_REGIMES}
    in_band = {precision: (sum(1 for point in predicted if point['precision'] == precision and point['band'][0] - 1e-9 <= point['speedup'] <= point['band'][1] + 1e-9),
                           sum(1 for point in predicted if point['precision'] == precision)) for precision, _, _ in PRECISIONS}
    check = validity(records)
    cell = lambda regime, layout, precision: (rf"{check[(regime, layout, precision)][2]}/{check[(regime, layout, precision)][0]} "
                                              rf"(${check[(regime, layout, precision)][1]:.2f}$)"
                                              if (regime, layout, precision) in check else "--")
    within = lambda regime, layout, precision: f"{check[(regime, layout, precision)][2]} of {check[(regime, layout, precision)][0]}"
    lines = []
    for regime, label, _ in LIMIT_REGIMES:
        for layout, layout_label, _ in LIMIT_LAYOUTS:
            lines.append(rf"{label if layout == 'r16' else ''} & {layout_label} & "
                         + " & ".join(cell(regime, layout, precision) for precision, _, _ in PRECISIONS) + r" \\")
    table_block = "\n".join(lines)
    domain = [point for point in points if point["regime"] == "domain"]
    wide = [point for point in domain if point["precision"] in ("fp64", "fp32")]
    misses = [point for point in wide if abs(point["rate_ratio"] - 1.0) > PREDICTION_TOLERANCE]
    hits = [point for point in wide if abs(point["rate_ratio"] - 1.0) <= PREDICTION_TOLERANCE]
    even = [point for point in points if point["imbalance"] <= IMBALANCE_LIMIT]
    tiny = [point for point in even if point["working_set"] < 0.5]
    tiny_time = median([point["time_us"] for point in tiny])
    tiny_speedup = median([point["rate_ratio"] / point["load_ratio"] for point in tiny])
    peak_point = max((point for point in even if point["resident"]), key=lambda point: point["rate_ratio"])
    peak = (peak_point["rate_ratio"], shared.tex_escape(peak_point["matrix"]))
    bracketed = sum(1 for point in domain if point["load_ratio"] <= point["rate_ratio"] <= 1.0)
    kept = {precision: median([(point["rate_ratio"] - point["load_ratio"]) / (1.0 - point["load_ratio"])
                               for point in domain if point["precision"] == precision and point["load_ratio"] < 0.97])
            for precision, _, _ in PRECISIONS}
    trend = spearman([point["load_ratio"] for point in domain], [point["rate_ratio"] for point in domain])
    slopes = {}
    for precision, _, _ in PRECISIONS:
        chosen = [point for point in domain if point["precision"] == precision]
        saved = [1.0 - point["load_ratio"] for point in chosen]
        lost = [1.0 - point["rate_ratio"] for point in chosen]
        slopes[precision] = sum(a * b for a, b in zip(saved, lost)) / sum(a * a for a in saved)
    domain_fp16 = median([check[("domain", layout, "fp16")][1] for layout in ("lp", "r16")])
    lowest_uneven = min(check[("uneven", layout, precision)][3] for layout in ("lp", "r16")
                        for precision, _, _ in PRECISIONS if ("uneven", layout, precision) in check)
    cache_range = [value for layout in ("lp", "r16") for precision, _, _ in PRECISIONS
                   if ("cache", layout, precision) in check
                   for value in (check[("cache", layout, precision)][3], check[("cache", layout, precision)][4])]
    spread_of = lambda precision: float(numpy.subtract(*numpy.percentile(
        [point["rate_ratio"] for point in domain if point["precision"] == precision and point["layout"] == "r16"], [75, 25])))
    spread_fp64, spread_fp16 = spread_of("fp64"), spread_of("fp16")
    campaign = {
        "cache": ("When both working sets fit in the L2, compression removes no DRAM traffic." "\n"
                  rf"Below half of the L2 a product takes a median of ${tiny_time:.0f}$~\textmu s with any layout, so the speedup is ${tiny_speedup:.2f}$ whatever the byte ratio and $K/K_{{\mathrm{{ref}}}}$ falls to $q/q_{{\mathrm{{ref}}}}$ (Fig.~\ref{{fig:limits}}A, top)." "\n"
                  rf"$K/K_{{\mathrm{{ref}}}}$ rises well above one only at the edge of the L2, up to ${peak[0]:.2f}$ on \texttt{{{peak[1]}}}, where the working set of the compressed layout still fits in the L2 and that of CSR-i32 no longer does."),
        "balance": rf"Over the campaign $K/K_{{\mathrm{{ref}}}}$ falls past the limit, as low as ${lowest_uneven:.2f}$ (Fig.~\ref{{fig:limits}}B, top).",
        "cache_brief": (rf"Below half of the L2 a product takes a median of ${tiny_time:.0f}$~\textmu s with any layout, so the speedup is ${tiny_speedup:.2f}$ whatever the byte ratio and $K/K_{{\mathrm{{ref}}}}$ falls to $q/q_{{\mathrm{{ref}}}}$ (Fig.~\ref{{fig:limits}}A)." "\n"
                        rf"$K/K_{{\mathrm{{ref}}}}$ rises well above one only at the edge of the L2, up to ${peak[0]:.2f}$ on \texttt{{{peak[1]}}}, where the working set of the compressed layout still fits in the L2 and that of CSR-i32 no longer does."),
        "lowest_uneven": f"{lowest_uneven:.2f}",
        "figure": rf"""\begin{{brief}}
\begin{{figure*}}[t]
  \centering
  \includegraphics[width=\textwidth]{{figures/limits_brief.pdf}}
  \caption{{The three factors. \textbf{{A}} $K/K_{{\mathrm{{ref}}}}$ of CSR-r16 and CSR-lp over the campaign on balanced matrices, one point per matrix, precision and layout, with a grey band ${100.0 * PREDICTION_TOLERANCE:.0f}$ per cent from one. \textbf{{B}} Active cycles of the busiest multiprocessor over the mean, for CSR-i32 on the profiled matrices outside the L2. \textbf{{C}} Share of the time the DRAM is busy against the DRAM bytes per load, both over CSR-i32, on the profiled balanced matrices outside the L2, with the dashed line where the two are equal.}}
  \label{{fig:limits}}
\end{{figure*}}
\end{{brief}}
\begin{{extended}}
\begin{{figure*}}[t]
  \centering
  \includegraphics[width=\textwidth]{{figures/limits.pdf}}
  \caption{{The three factors against their indicators. Top row, $K/K_{{\mathrm{{ref}}}}$ of CSR-r16 and CSR-lp over the campaign, one point per matrix, precision and layout, each panel on the matrices that meet the other two conditions, with a grey band ${100.0 * PREDICTION_TOLERANCE:.0f}$ per cent from one and, in \textbf{{C}}, the dashed line $K/K_{{\mathrm{{ref}}}} = q/q_{{\mathrm{{ref}}}}$ of a speedup of one. Bottom row, the counter that explains each factor on the profiled matrices, \textbf{{A}} the bytes the DRAM delivers over $Q$, \textbf{{B}} the active cycles of the busiest multiprocessor over the mean, \textbf{{C}} the share of the time the DRAM is busy over that of CSR-i32 against the DRAM bytes per load over CSR-i32.}}
  \label{{fig:limits}}
\end{{figure*}}
\end{{extended}}
""",
        "flight": (rf"On balanced matrices outside the L2 the speedup grows as the values narrow, yet it reaches a median ${kept['fp64']:.2f}$, ${kept['fp32']:.2f}$ and ${kept['fp16']:.2f}$ of the gain $q_{{\mathrm{{ref}}}}/q - 1$ that the byte ratio would give at an unchanged rate, at FP64, FP32 and FP16, and $K/K_{{\mathrm{{ref}}}}$ of CSR-r16 varies more from matrix to matrix, with an interquartile range of ${spread_fp64:.2f}$ at FP64 and ${spread_fp16:.2f}$ at FP16, so the more the index weighs in the bytes, the less of its saving reaches the time and the less the byte ratio alone tells about a single matrix." "\n"
                   rf"\extonly{{The loss is not a fixed fraction of the saving, since fitting $1-K/K_{{\mathrm{{ref}}}} = k\,(1-q/q_{{\mathrm{{ref}}}})$ gives $k = {slopes['fp64']:.2f}$ at FP64, ${slopes['fp32']:.2f}$ at FP32 and ${slopes['fp16']:.2f}$ at FP16.}}"),
    }
    value_width = {key: size for key, _, size in PRECISIONS}
    facts = {(record["matrix"], record["precision"]): record for record in records}
    positions, rereads, rates = [], [], []
    for entry in measurements:
        record = facts.get((entry["matrix"], entry["precision"]))
        if entry["layout"] != "r16" or record is None or regime_of(record) != "domain" or "r16" not in record["layouts"]:
            continue
        vector = record["bytes"] / record["q"] / record["row_length"] * value_width[entry["precision"]]
        rereads.append((record["bytes"] / entry["counted_reference"] - (record["bytes"] - 2.0 * vector)) / vector)
        positions.append(band_position(record["layouts"]["r16"]))
        rates.append(record["layouts"]["r16"]["speedup"] / record["layouts"]["r16"]["byte_ratio"])
    rereads_rho, rate_rho = (spearman(positions, rereads), spearman(positions, rates)) if positions else (0.0, 0.0)
    pooled = lambda precision: (sum(check[("domain", layout, precision)][2] for layout in ("lp", "r16")),
                                sum(check[("domain", layout, precision)][0] for layout in ("lp", "r16")))
    rest = rf"""
\begin{{brief}}
On balanced matrices outside the L2 the byte ratio predicts the speedup within ${100.0 * PREDICTION_TOLERANCE:.0f}$ per cent on {pooled('fp64')[0]} of the {pooled('fp64')[1]} combinations of matrix and layout at FP64, {pooled('fp32')[0]} of {pooled('fp32')[1]} at FP32 and only {pooled('fp16')[0]} of {pooled('fp16')[1]} at FP16.
\end{{brief}}
\begin{{extended}}
Where neither the cache nor the balance of the rows acts, on balanced matrices outside the L2, the byte ratio predicts the speedup within ${100.0 * PREDICTION_TOLERANCE:.0f}$ per cent on {within('domain', 'lp', 'fp64')} matrices for CSR-lp and {within('domain', 'r16', 'fp64')} for CSR-r16 at FP64, and on {within('domain', 'lp', 'fp32')} and {within('domain', 'r16', 'fp32')} at FP32.
On all {len(misses)} matrices it misses at FP64 and FP32 the byte ratio overestimates the speedup, with $K/K_{{\mathrm{{ref}}}}$ between ${min(point['rate_ratio'] for point in misses):.2f}$ and ${max(point['rate_ratio'] for point in misses):.2f}$, and they lie at the edges of the first and third conditions, with a median working set of ${median([point['working_set'] for point in misses]):.1f}$ times the L2 against ${median([point['working_set'] for point in hits]):.1f}$ for the matrices it predicts and a median $q/q_{{\mathrm{{ref}}}}$ of ${median([point['load_ratio'] for point in misses]):.2f}$ against ${median([point['load_ratio'] for point in hits]):.2f}$.
At FP16 it does so on {within('domain', 'lp', 'fp16')} and {within('domain', 'r16', 'fp16')}, and the speedup falls short of it by a median ${100.0 * (1.0 - domain_fp16):.0f}$ per cent, a median $K/K_{{\mathrm{{ref}}}}$ of ${domain_fp16:.2f}$, the third factor.
Outside the L2 the DRAM delivers about the bytes $Q$, so the excess of the gather is close to zero and the real byte ratio is the upper end of the band of Eq.~\eqref{{eq:band}} rather than a point inside it.
The lower end, which charges a whole transaction to every gather, is the most pessimistic count of the bytes, and {below['uneven'][0]} of the {below['uneven'][1]} combinations of matrix, precision and layout on unbalanced matrices fall below it, so their loss comes from the rate of the kernel and not from its bytes\extonly{{ (Fig.~\ref{{fig:prediction}})}}.
Inside the band the position of a speedup does not follow the rereads of $x$, with a rank correlation of ${rereads_rho:+.2f}$ with them for CSR-r16 on the profiled balanced matrices against ${rate_rho:+.2f}$ with $K/K_{{\mathrm{{ref}}}}$, so the band bounds the bytes and does not explain the speedup.
In the L2 the band does not apply, since the cache serves part of the matrix and of $x$ differently for each layout, and {below['cache'][0]} of the {below['cache'][1]} combinations there fall below its lower end as well.
The byte ratio is therefore a prediction within ${100.0 * PREDICTION_TOLERANCE:.0f}$ per cent on balanced matrices at FP64 and FP32 and an upper estimate at FP16, while on unbalanced matrices and in the L2 $K/K_{{\mathrm{{ref}}}}$ departs from one in either direction and Eq.~\eqref{{eq:srate}} relates the speedup to the byte ratio only through it.
\end{{extended}}

\begin{{extended}}
\begin{{table}}[t]
\centering
\caption{{Matrices on which the byte ratio predicts the speedup within ${100.0 * PREDICTION_TOLERANCE:.0f}$ per cent, out of those on which the layout is exact, and in parentheses the median $K/K_{{\mathrm{{ref}}}}$, per regime, after RCM.}}
\label{{tab:results-limits}}
\footnotesize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{llccc}}
\toprule
regime & layout & FP64 & FP32 & FP16 \\
\midrule
{table_block}
\bottomrule
\end{{tabular}}
\end{{table}}
\end{{extended}}


\begin{{extended}}
\begin{{figure*}}[t]
  \centering
  \includegraphics[width=\textwidth]{{figures/prediction.pdf}}
  \caption{{Measured speedup over CSR-i32 of CSR-r16 and CSR-lp (markers) and gather band of Eq.~\eqref{{eq:band}} (grey bars), one row per regime and one column per precision, matrices sorted by the byte ratio $q_{{\mathrm{{ref}}}}/q$. The top of each bar is the byte ratio, which the measured traffic matches outside the L2, and the bottom charges a whole transaction to every gather, so a marker below its bar, in black, loses to the rate of the kernel more than any count of the bytes allows. Each panel has its own speedup axis.}}
  \label{{fig:prediction}}
\end{{figure*}}
\end{{extended}}
"""
    return campaign, rest


# Unrolling factors of the inner loop at FP16 with FP32 accumulation, read with cuobjdump from the
# sm_80 binary of the campaign: nvcc unrolls CSR-r16 by 4 and CSR-i32 (and every FP64/FP32 kernel) by 16.
UNROLL_R16_FP16, UNROLL_I32 = 4, 16


def fp16_controls(controls_dir, device, characterization):
    """The FP16 campaign repeated with three builds: the default one, one that accumulates in FP16 and
    one that unrolls the loop of CSR-r16 by 16. Returns, per build, the geometric mean K/K_ref of
    CSR-r16 on balanced matrices outside the L2."""
    builds = {"default": "bench_csr_width", "fp16acc": "bench_csr_width_fp16acc",
              "unroll": "bench_csr_width_r16unroll"}
    if controls_dir is None or not all((controls_dir / binary / "csr_layouts_rcm.csv").is_file()
                                       for binary in builds.values()):
        return None
    runs = {}
    for build, binary in builds.items():
        csv_path = controls_dir / binary / "csr_layouts_rcm.csv"
        table, _ = at_reference_threads(shared.load_measurements(csv_path),
                                        load_thread_samples(csv_path.with_name(csv_path.stem + "_tpr.csv")))
        runs[build] = {record["matrix"]: record
                       for record in of_precision(summarize(table, device, characterization), "fp16")}
    common = sorted(set.intersection(*(set(run) for run in runs.values())))
    base = runs["default"]
    even = [name for name in common if regime_of(base[name]) == "domain"
            and all("r16" in runs[build][name]["layouts"] for build in builds)]
    result = {"matrices": len(common), "even": len(even)}
    for build in builds:
        entries = [runs[build][name]["layouts"]["r16"] for name in even]
        result[build] = geometric_mean([entry["speedup"] / entry["byte_ratio"] for entry in entries])
    return result


def second_device(campaign_dir, characterization, profile_dir=None, rate_list=None):
    """The same campaign on a second GPU. Returns its description and, per precision, the number of
    matrices with even rows outside its L2 on which CSR-r16 is exact, the geometric mean K/K_ref, and
    the number on which the byte ratio predicts the speedup within PREDICTION_TOLERANCE."""
    if campaign_dir is None or not (campaign_dir / "csr_layouts_rcm.csv").is_file():
        return None
    csv_path = campaign_dir / "csr_layouts_rcm.csv"
    device = load_device(csv_path, campaign_dir / "device.txt")
    device.read_bw_gbs = device.theoretical_bw_gbs
    table, _ = at_reference_threads(shared.load_measurements(csv_path),
                                    load_thread_samples(campaign_dir / "csr_layouts_rcm_tpr.csv"))
    records = summarize(table, device, characterization)
    result = {"name": device.name, "arch": device.arch, "bandwidth": device.theoretical_bw_gbs,
              "l2": device.l2_mib, "matrices": len(of_precision(records, "fp64")),
              "resident": sum(1 for record in of_precision(records, "fp16") if record["resident"])}
    for precision, _, _ in PRECISIONS:
        chosen = [record["layouts"]["r16"] for record in with_layout(of_precision(records, precision), "r16")
                  if regime_of(record) == "domain"]
        ratios = [entry["speedup"] / entry["byte_ratio"] for entry in chosen]
        result[precision] = (len(chosen), geometric_mean(ratios),
                             sum(1 for ratio in ratios if abs(ratio - 1.0) <= PREDICTION_TOLERANCE))
        group = [record for record in with_layout(of_precision(records, precision), "r16") if regime_of(record) == "domain"]
        result.setdefault("table", {})[precision] = (
            geometric_mean([record["q"] for record in group]), geometric_mean([record["layouts"]["r16"]["q"] for record in group]),
            geometric_mean([record["rate"] for record in group]), geometric_mean([record["layouts"]["r16"]["rate"] for record in group]))
    if profile_dir is not None and profile_dir.is_dir():
        profiled = [entry for entry in rate_measurements(profile_dir, rate_list, csv_path)
                    if entry["group"] == "domain" and entry["precision"] == "fp16"]
        result["profile"] = {layout: {key: median([entry[key] for entry in profiled if entry["layout"] == layout])
                                      for key in ("per_load", "occupancy_reference", "occupancy_layout",
                                                  "busy_reference", "busy_layout")}
                             for layout in ("r16", "lp")}
    return result


def cusparse_profile(study_dir, characterization):
    """Nsight Compute profile of cuSPARSE beside CSR-i32 on a few matrices. Every timed call of
    cusparseSpMV runs a kernel that scales y before its SpMV kernel. Returns the range of the scaling
    kernel in microseconds and, on the matrices with even rows, the range of the SpMV kernel of the
    faster algorithm over the fastest CSR-i32, at FP64 and FP32 together and at FP16."""
    if study_dir is None or not study_dir.is_dir():
        return None
    type_names = {"double": "fp64", "float": "fp32", "__half": "fp16"}
    scale_times, ratios, matrices = [], {"wide": [], "fp16": []}, []
    for path in sorted(study_dir.glob("*.ncu.csv")):
        matrix = path.name[:-len(".ncu.csv")]
        if matrix not in characterization:
            continue
        rows = shared.read_rows(path)[1:]
        even = (int(characterization[matrix]["row_length_max"])
                / float(characterization[matrix]["row_length_mean"])) <= IMBALANCE_LIMIT
        blocks, current = [], None
        for row in rows:
            name, duration = row["Kernel Name"], float(row["gpu__time_duration.sum"]) / 1e3
            if name.startswith("k_read_bw"):
                continue
            if name.startswith("void k_csr"):
                precision = type_names[name.split("<")[1].split(",")[0]]
                if current is None or current["precision"] != precision or current["vendor"]:
                    current = {"precision": precision, "own": [], "vendor": []}
                    blocks.append(current)
                current["own"].append((name, duration))
            elif current is not None:
                current["vendor"].append((name.split("::")[1].split("<")[0], duration))
        matrices.append(matrix)
        for block in blocks:
            reference = min(duration for name, duration in block["own"]
                            if name.startswith("void k_csr_cumulative") and ", int," in name)
            calls = []
            for name, duration in block["vendor"]:
                if name == "csrmv_v3_partition_kernel":
                    calls.append([])
                elif calls:
                    calls[-1].append((name, duration))
            timed = []
            for algorithm in calls:
                last = max(index for index, (name, _) in enumerate(algorithm)
                           if name == "vector_scalar_multiply_kernel")
                call = algorithm[last:]
                if even:
                    scale_times.extend(duration for name, duration in call if name == "vector_scalar_multiply_kernel")
                timed.append((sum(duration for _, duration in call),
                               next(duration for name, duration in call if name == "csrmv_v3_kernel")))
            if even and timed:
                ratios["fp16" if block["precision"] == "fp16" else "wide"].append(min(timed)[1] / reference)
    if not ratios["wide"] or not ratios["fp16"]:
        return None
    return {"matrices": len(matrices), "even": len(ratios["fp16"]),
            "scale": (min(scale_times), max(scale_times)),
            "wide": (min(ratios["wide"]), max(ratios["wide"])),
            "fp16": (min(ratios["fp16"]), max(ratios["fp16"]))}


def rate_tex(measurements, records, ordering, campaign, controls=None, second=None):
    if not measurements:
        return ""
    device_cuda = shared.DEVICE.cuda if getattr(shared.DEVICE, 'cuda', None) else '12'
    ordering_text = ("after the same reordering as the campaign" if ordering == "rcm"
                     else "in the original ordering of the matrix")
    pick = lambda group, layout, precision=None: [entry for entry in measurements
                                                   if entry["group"] == group and entry["layout"] == layout
                                                   and (precision is None or entry["precision"] == precision)]
    matrices = lambda group: len({entry["matrix"] for entry in measurements if entry["group"] == group})
    middle = lambda group, layout, key, precision=None: median([entry[key] for entry in pick(group, layout, precision)])
    ratio = lambda group, layout, precision: middle(group, layout, "rate_ratio", precision)
    lines = []
    for group, label in RATE_GROUPS:
        for layout, layout_label in (("lp", "CSR-lp"), ("r16", "CSR-r16")):
            entries = pick(group, layout)
            if not entries:
                continue
            lines.append(rf"{label if layout == 'lp' else ''} & {matrices(group) if layout == 'lp' else ''} & {layout_label} & "
                         rf"${median([1.0 / entry['counted_layout'] for entry in entries]):.2f}$ & "
                         rf"${median([entry['spread'] for entry in entries]):.2f}$ & "
                         rf"${middle(group, layout, 'per_load', 'fp64'):.2f}$ & "
                         rf"${middle(group, layout, 'per_load', 'fp16'):.2f}$ \\")
    table_block = "\n".join(lines)
    domain_pairs = [(entry, entry["counted_reference"]) for entry in pick("domain", "lp") + pick("domain", "r16")] + \
                   [(entry, entry["counted_layout"]) for entry in pick("domain", "lp") + pick("domain", "r16")]
    counted_median = median([value for _, value in domain_pairs])
    exceptions = sorted({(entry["matrix"], entry["precision"], entry["layout"], round(value, 3))
                         for entry, value in domain_pairs if abs(value - 1.0) > COUNT_TOLERANCE})
    layout_name = {"lp": "CSR-lp", "r16": "CSR-r16"}
    if len(exceptions) == 1:
        matrix, precision, layout, value = exceptions[0]
        exception_text = (rf", except for {layout_name[layout]} on \texttt{{{shared.tex_escape(matrix)}}} at {precision.upper()}, "
                          rf"which reads ${100.0 * abs(1.0 / value - 1.0):.0f}$ per cent {'more' if value < 1.0 else 'less'} than its count")
    elif exceptions:
        names = sorted({matrix for matrix, _, _, _ in exceptions})
        exception_text = (rf" on all but {len(exceptions)} of the {len(domain_pairs) // 2} profiled pairs, on "
                          + ", ".join(rf"\texttt{{{shared.tex_escape(name)}}}" for name in names)
                          + rf", which depart by at most ${100.0 * max(abs(1.0 / value - 1.0) for _, _, _, value in exceptions):.0f}$ per cent")
    else:
        exception_text = ""
    gather_counted = median([entry["counted_reference"] for entry in pick("gather", "lp")])
    gather_text = (rf"the DRAM delivers a median ${100.0 * abs(1.0 - 1.0 / gather_counted):.0f}$ per cent "
                   + ("less than the count of CSR-i32, since the reordering lets the L2 keep part of $x$"
                      if gather_counted > 1.0 else "more than the count of CSR-i32, the cost of the gather"))
    exact_matrices = len({entry["matrix"] for entry in measurements if entry["layout"] == "r16"})
    exact_text = rf"CSR-r16 is exact on {exact_matrices} of them"
    uneven_exact = len({entry["matrix"] for entry in pick("uneven", "r16")})
    uneven_relative = ("\n" + (rf"On the one of them on which CSR-r16 is exact" if uneven_exact == 1
                               else rf"On the {uneven_exact} of them on which CSR-r16 is exact")
                       + rf" it falls only to ${ratio('uneven', 'r16', 'fp16'):.2f}$ at FP16, which points to the decoding of CSR-lp on the chain of the longest rows."
                       if uneven_exact else "")
    cache_ratio = [entry["rate_ratio"] for entry in pick("cache", "lp") + pick("cache", "r16")]
    flight_rows, flight_lines, flight_sizes = {}, [], []
    for precision, precision_label, _ in PRECISIONS:
        group = [record for record in with_layout(of_precision(records, precision), "r16") if regime_of(record) == "domain"]
        flight_sizes.append(str(len(group)))
        q_ref = geometric_mean([record["q"] for record in group])
        q_new = geometric_mean([record["layouts"]["r16"]["q"] for record in group])
        k_ref = geometric_mean([record["rate"] for record in group])
        k_new = geometric_mean([record["layouts"]["r16"]["rate"] for record in group])
        byte_ratio = geometric_mean([record["layouts"]["r16"]["byte_ratio"] for record in group])
        rate_ratio = geometric_mean([record["layouts"]["r16"]["speedup"] / record["layouts"]["r16"]["byte_ratio"] for record in group])
        speedup = geometric_mean([record["layouts"]["r16"]["speedup"] for record in group])
        flight_lines.append(rf"A100 {precision_label} & ${q_ref:.1f}$ & ${q_new:.1f}$ & ${k_ref:.0f}$ & ${k_new:.0f}$ \\")
        flight_rows[precision] = (q_ref, q_new, k_ref, k_new, byte_ratio, rate_ratio, speedup)
    f64, f16 = flight_rows["fp64"], flight_rows["fp16"]
    flight_rows["fp64"] = (rf"At FP64 CSR-i32 moves ${f64[0]:.1f}$ bytes per nonzero and CSR-r16 ${f64[1]:.1f}$, both run at about the same rate, ${f64[2]:.0f}$ and ${f64[3]:.0f}$~GB/s, "
                           rf"and the speedup of ${f64[6]:.2f}$ is close to the byte ratio of ${f64[4]:.2f}$.")
    flight_rows["fp16"] = (rf"at FP16, with ${f16[0]:.1f}$ and ${f16[1]:.1f}$ bytes per nonzero, CSR-r16 runs at ${f16[3]:.0f}$~GB/s against ${f16[2]:.0f}$, "
                           rf"$K/K_{{\mathrm{{ref}}}} = {f16[5]:.2f}$, and the speedup is ${f16[6]:.2f}$ against a byte ratio of ${f16[4]:.2f}$")
    if second:
        flight_lines.append(r"\midrule")
        for precision, precision_label, _ in PRECISIONS:
            row = second["table"][precision]
            flight_lines.append(rf"Quadro {precision_label} & ${row[0]:.1f}$ & ${row[1]:.1f}$ & ${row[2]:.0f}$ & ${row[3]:.0f}$ \\")
    flight_table = "\n".join(flight_lines)
    flight_counts = ", ".join(flight_sizes[:-1]) + " and " + flight_sizes[-1]
    cache_counted = max(entry["counted_layout"] for entry in pick("cache", "lp") + pick("cache", "r16"))
    real_gap = median([abs(entry["counted_layout"] / entry["counted_reference"] - 1.0) for entry in measurements
                       if entry["group"] != "cache"])
    even_fp16 = [record for record in with_layout(of_precision(records, "fp16"), "r16") if regime_of(record) == "domain"]
    size_correlation = numpy.corrcoef(numpy.log10([record["working_set"] for record in even_fp16]),
                                      [record["layouts"]["r16"]["speedup"] / record["layouts"]["r16"]["byte_ratio"]
                                       for record in even_fp16])[0, 1]
    size_text = (rf"$K/K_{{\mathrm{{ref}}}}$ of CSR-r16 at FP16 on these {len(even_fp16)} matrices has a correlation of only ${size_correlation:.2f}$ "
                 "with the logarithm of the working set, so the loss does not shrink as the matrix grows")
    if controls:
        controls_text = (
            r"\begin{brief}" "\n"
            rf"Repeating the FP16 measurements on all {controls['matrices']} matrices with the products accumulated in FP16 rather than FP32 gives $K/K_{{\mathrm{{ref}}}} = {controls['fp16acc']:.2f}$ against ${controls['default']:.2f}$, so the loss does not lie in the precision of the accumulation." "\n"
            r"\end{brief}" "\n"
            r"\begin{extended}" "\n"
            "Three further measurements show that the loss does not depend on the precision of the accumulation, on the unrolling of the loop or on the size of the matrix." "\n"
            rf"The FP16 measurements were repeated on all {controls['matrices']} matrices with three builds of the kernels, and the default one gives $K/K_{{\mathrm{{ref}}}} = {controls['default']:.2f}$ for CSR-r16 on the {controls['even']} balanced matrices outside the L2, as in the campaign." "\n"
            rf"Accumulating the products in FP16 rather than in FP32, which removes the conversion of every product to FP32, gives ${controls['fp16acc']:.2f}$." "\n"
            rf"At FP16 the compiler unrolls the loop of CSR-r16 by a factor of {UNROLL_R16_FP16} and that of CSR-i32 by {UNROLL_I32}, and unrolling CSR-r16 by {UNROLL_I32} as well gives ${controls['unroll']:.2f}$, so the shorter unrolling does not cause the loss." "\n"
            rf"Finally, {size_text}, as it would if it were a fixed cost of each launch rather than a lower rate held over the whole product." "\n"
            r"\end{extended}" "\n")
    else:
        controls_text = ""
    if second:
        a100_bandwidth = shared.DEVICE.theoretical_bw_gbs
        controls_text += (
            rf"The same campaign on a {second['name']} ({second['arch']}, ${second['bandwidth']:.0f}$~GB/s, {second['l2']:.0f}~MiB L2), "
            rf"whose L2 is smaller than every working set, shows the same pattern." "\n"
            rf"On the {second['fp64'][0]} balanced matrices on which CSR-r16 is exact, its $K/K_{{\mathrm{{ref}}}}$ is ${second['fp64'][1]:.2f}$, ${second['fp32'][1]:.2f}$ and ${second['fp16'][1]:.2f}$ at FP64, FP32 and FP16, "
            rf"and the byte ratio predicts the speedup within ${100.0 * PREDICTION_TOLERANCE:.0f}$ per cent on {second['fp64'][2]}, {second['fp32'][2]} and {second['fp16'][2]} of them." "\n"
            rf"Its counters show the same trend for CSR-r16 at FP16, with ${second['profile']['r16']['per_load']:.2f}$ of the bytes per load of CSR-i32 at the same occupancy and a DRAM busy share that falls from ${second['profile']['r16']['busy_reference']:.0f}$ to ${second['profile']['r16']['busy_layout']:.0f}$ per cent, "
            rf"while CSR-lp, with ${second['profile']['lp']['per_load']:.2f}$ of the bytes per load, keeps it at ${second['profile']['lp']['busy_layout']:.0f}$ per cent, so on this device the bytes per load do not set the loss alone." "\n")
    return rf"""
\subsection{{Characterization of the rate ratio}}
\label{{sec:results-rate}}
\label{{sec:results-limits}}

The causes of the second factor of Eq.~\eqref{{eq:identity}} are read from the hardware counters that Nsight Compute collects for each kernel.
CSR-i32, CSR-r16 and CSR-lp are profiled on {len({entry['matrix'] for entry in measurements})} of the matrices, chosen in four groups so that each group departs from the first in one respect only.
The {matrices('domain')} matrices of the first group are balanced with a working set above the L2, the longest row at most {IMBALANCE_LIMIT:g} times the mean, {matrices('gather')} have short rows and an $x$ larger than the L2, {matrices('uneven')} are less balanced than that limit, and {matrices('cache')} fit in the L2.
Each configuration is profiled once, {ordering_text} and at the $T$ that is fastest for CSR-i32, without flushing the caches between the replays of the profiler, so that a matrix in the L2 is measured warm as in the campaign.
{exact_text}.

\begin{{extended}}
The profiler is the Nsight Compute of CUDA {device_cuda}, run with the options \texttt{{--target-processes all}}, \texttt{{--kernel-name-base demangled}} and \texttt{{--cache-control none}}, the metrics of Table~\ref{{tab:ncu-metrics}} and a filter on the names of the three kernels, in its default kernel replay and with the clocks locked to their base, its default.
Each configuration is launched once, with no warm-up and no repetition, so the profiler times one product, and only ratios between kernels profiled under the same conditions are used.
The counters also collected, the requests and sectors of the L1 and of the L2 and the remaining stall reasons, are not used here.
\begin{{table*}}[t]
\centering
\caption{{Metrics of Nsight Compute read in Sec.~\ref{{sec:results-limits}}, with what each measures and the factor of Fig.~\ref{{fig:limits}} it explains.}}
\label{{tab:ncu-metrics}}
\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{llc}}
\toprule
metric & quantity & factor \\
\midrule
\texttt{{dram\_\_bytes\_read.sum}} & bytes the DRAM delivers, against the count $Q$ & A \\
\texttt{{gpu\_\_time\_duration.sum}} & time of the profiled kernel, for $K$ of the profiled runs & all \\
\texttt{{sm\_\_cycles\_active.max, .avg}} & busiest multiprocessor over the mean & B \\
\texttt{{dram\_\_throughput.avg.pct\_of\_peak\_sustained\_elapsed}} & share of the time the DRAM is busy & C \\
\texttt{{smsp\_\_inst\_executed\_op\_global\_ld.sum}} & global load instructions, for the DRAM bytes per load & C \\
\texttt{{smsp\_\_inst\_executed.sum}} & instructions executed, for the work a layout adds & C \\
\texttt{{sm\_\_warps\_active.avg.pct\_of\_peak\_sustained\_active}} & achieved occupancy & C \\
\texttt{{smsp\_\_warp\_issue\_stalled\_long\_scoreboard\_per\_warp\_active.pct}} & warp cycles waiting on global memory & C \\
\texttt{{smsp\_\_warp\_issue\_stalled\_wait\_per\_warp\_active.pct}} & warp cycles waiting on a fixed-latency result & C \\
\bottomrule
\end{{tabular}}
\end{{table*}}
\end{{extended}}
Since $K$ is $Q$ over the time and the DRAM delivers its own bytes in the same time, $K/K_{{\mathrm{{ref}}}}$ is the ratio of the busy shares of the DRAM of the two kernels, corrected by how far each count departs from the real traffic.
Three factors move it away from one, the working set against the L2, the longest row against the mean, and the bytes a load carries relative to CSR-i32, approximated by $q/q_{{\mathrm{{ref}}}}$, which it equals for CSR-r16 since both issue three loads per nonzero.
These quantities tell where $K/K_{{\mathrm{{ref}}}}$ departs from one and in which direction, but not its value on a single matrix, which also depends on the decoding of CSR-lp and on how close the matrix lies to the limit of a regime.
\briefonly{{Fig.~\ref{{fig:limits}} gives the effect of the cache on $K/K_{{\mathrm{{ref}}}}$ over the campaign and the counters behind the other two factors on the profiled matrices.}}
\extonly{{Fig.~\ref{{fig:limits}} gives each factor in one column, with the measured $K/K_{{\mathrm{{ref}}}}$ against its indicator in the top row and the hardware counter that shows the mechanism on the profiled matrices in the bottom row, and Table~\ref{{tab:results-rate}} gives the counters per group.}}

{campaign['figure']}
\paragraph*{{A. Cache}}
Outside the L2 the ratio of the bytes the DRAM delivers to CSR-i32 and to a compressed layout (\texttt{{dram\_\_bytes\_read}}) differs from $q_{{\mathrm{{ref}}}}/q$ by a median of ${100.0 * real_gap:.1f}$ per cent, so the byte ratio is the ratio of the real traffic, while in the L2 the DRAM delivers as little as ${100.0 / cache_counted:.0f}$ per cent of the bytes $Q$ and the time no longer follows the bytes\extonly{{ (Fig.~\ref{{fig:limits}}A, bottom)}}.
\begin{{brief}}
{campaign['cache_brief']}
\end{{brief}}
\begin{{extended}}
The counter sees only reads, so on the matrices with short rows the DRAM delivers less than $Q$ by the writes of $y$, while $K/K_{{\mathrm{{ref}}}}$ stays at ${ratio('gather', 'lp', 'fp64'):.2f}$.
{campaign['cache']}
\end{{extended}}

\paragraph*{{B. Balance of the rows}}
\begin{{brief}}
The active cycles of the multiprocessors (\texttt{{sm\_\_cycles\_active}}) stay even up to the limit of {IMBALANCE_LIMIT:g} on the longest row, and past it the busiest multiprocessor runs up to ${max(entry['spread'] for entry in pick('uneven', 'lp')):.1f}$ times the mean while the DRAM is busy for only ${middle('uneven', 'lp', 'busy_reference', 'fp64'):.0f}$ per cent of the time, so the chain of the longest rows sets the time and $K/K_{{\mathrm{{ref}}}}$ falls as low as ${campaign['lowest_uneven']}$ (Fig.~\ref{{fig:limits}}B).
\end{{brief}}
\begin{{extended}}
The active cycles of the multiprocessors (\texttt{{sm\_\_cycles\_active}}) stay even up to the limit of {IMBALANCE_LIMIT:g} on the longest row, and past it the busiest multiprocessor runs up to ${max(entry['spread'] for entry in pick('uneven', 'lp')):.1f}$ times the mean while the DRAM is busy for only ${middle('uneven', 'lp', 'busy_reference', 'fp64'):.0f}$ per cent of the time (Fig.~\ref{{fig:limits}}B, bottom).
The time is then the chain of steps of the longest rows, which compression does not shorten.
{campaign['balance']}
The loss is larger for CSR-lp, ${ratio('uneven', 'lp', 'fp16'):.2f}$ against ${ratio('uneven', 'r16', 'fp16'):.2f}$ for CSR-r16 at FP16 on the profiled matrices, consistent with the cost of its decoding, which on that chain is no longer hidden behind the memory accesses.
\end{{extended}}

\paragraph*{{C. Bytes in flight}}
Since the byte ratio is the ratio of the real traffic outside the L2, on balanced matrices the gap between the speedup and the byte ratio is a gap in rate.
Table~\ref{{tab:rate-bytes}} gives the bytes per nonzero and the rate of CSR-i32 and CSR-r16 on these matrices.
{flight_rows['fp64']}
As the values narrow, CSR-i32 itself moves fewer bytes per nonzero, and the two bytes that CSR-r16 removes from each of them cost a larger share of its rate, so that {flight_rows['fp16']}.
The counters show why on the profiled matrices.
CSR-r16 adds one integer addition per nonzero to CSR-i32, keeps its occupancy, ${middle('domain', 'r16', 'occupancy_layout', 'fp16'):.0f}$ against ${middle('domain', 'r16', 'occupancy_reference', 'fp16'):.0f}$ per cent at FP16 (\texttt{{sm\_\_warps\_active}}), and issues the same three loads per nonzero, but each load carries fewer DRAM bytes, ${middle('domain', 'r16', 'per_load', 'fp64'):.2f}$ of those of CSR-i32 at FP64 and ${middle('domain', 'r16', 'per_load', 'fp16'):.2f}$ at FP16 (\texttt{{dram\_\_bytes\_read}} over \texttt{{smsp\_\_inst\_executed\_op\_global\_ld}}).
It executes ${middle('domain', 'r16', 'instructions', 'fp64'):.2f}$ and ${middle('domain', 'r16', 'instructions', 'fp16'):.2f}$ times the instructions of CSR-i32 at FP64 and FP16 (\texttt{{smsp\_\_inst\_executed}}), so its added work does not grow as the values narrow.
By Little's law~\cite{{little1961}} the memory stays busy only while enough bytes are in flight to cover its latency, so at the same occupancy loads with fewer bytes each leave the DRAM idle for part of it, and its busy share (\texttt{{dram\_\_throughput}}) stays at ${middle('domain', 'r16', 'busy_layout', 'fp64'):.0f}$ per cent at FP64 but falls from ${middle('domain', 'r16', 'busy_reference', 'fp16'):.0f}$ to ${middle('domain', 'r16', 'busy_layout', 'fp16'):.0f}$ per cent at FP16 (Fig.~\ref{{fig:limits}}C).
{controls_text}CSR-r16 keeps the loads of CSR-i32 by design, to isolate the effect of the bytes, so this loss measures how many bytes each load of a compressed format must carry back, with wider loads or more nonzeros per thread, to turn the whole byte ratio into time.

\begin{{table}}[t]
\centering
\caption{{Bytes per nonzero $q$ and rate $K$ of CSR-i32 and CSR-r16 on balanced matrices outside the L2, geometric means over the {flight_counts} matrices on which CSR-r16 is exact at FP64, FP32 and FP16 on the A100, whose ratios are those of Table~\ref{{tab:results-gain}}, and over {second['fp64'][0] if second else 0} on the Quadro.}}
\label{{tab:rate-bytes}}
\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lrrrr}}
\toprule
 & \multicolumn{{2}}{{c}}{{$q$ (B)}} & \multicolumn{{2}}{{c}}{{$K$ (GB/s)}} \\
\cmidrule(lr){{2-3}}\cmidrule(lr){{4-5}}
 & CSR-i32 & CSR-r16 & CSR-i32 & CSR-r16 \\
\midrule
{flight_table}
\bottomrule
\end{{tabular}}
\end{{table}}
{campaign['flight']}

\begin{{extended}}
\begin{{table}}[t]
\centering
\caption{{Nsight Compute counters per profiled group, medians: DRAM bytes over $Q$, busiest multiprocessor over the mean, DRAM bytes per load over CSR-i32.}}
\label{{tab:results-rate}}
\footnotesize
\setlength{{\tabcolsep}}{{2.2pt}}
\begin{{tabular}}{{llcrrrr}}
\toprule
 & & & & & \multicolumn{{2}}{{c}}{{B/load}} \\
\cmidrule(lr){{6-7}}
group & $n$ & layout & $\frac{{Q_{{\mathrm{{DRAM}}}}}}{{Q}}$ & SM max & FP64 & FP16 \\
\midrule
{table_block}
\bottomrule
\end{{tabular}}
\end{{table}}
\end{{extended}}
"""


def bound_check(records):
    """The texts that report the bounds of Eq. sbound and Eq. slayout against the measured speedups,
    with the median fraction of its own bound that CSR-r16 reaches at each precision."""
    pairs, layout_above, layout_above_resident = 0, 0, 0
    above, above_resident = 0, 0
    beyond_ceiling = []
    capture = {}
    for precision, _, _ in PRECISIONS:
        for record in of_precision(records, precision):
            for layout, measured in record["layouts"].items():
                pairs += 1
                if measured["speedup"] > measured["layout_bound"]:
                    layout_above += 1
                    layout_above_resident += record["resident"]
                if measured["speedup"] > record["bound"]:
                    above += 1
                    above_resident += record["resident"]
                if measured["speedup"] > record["r_max"]:
                    beyond_ceiling.append((record, dict(LAYOUTS)[layout]))
        capture[precision] = median([record["layouts"]["r16"]["speedup"] / record["layouts"]["r16"]["layout_bound"]
                                     for record in with_layout(of_precision(records, precision), "r16")])
    if layout_above == 0:
        layout_text = rf"None of the {pairs} measured pairs of matrix and layout exceeds the bound of its own layout, Eq.~\eqref{{eq:slayout}}"
    elif layout_above == layout_above_resident:
        layout_text = (rf"No speedup on a matrix served from the DRAM exceeds the bound of its own layout, Eq.~\eqref{{eq:slayout}}, "
                       rf"and the {layout_above} of the {pairs} measured pairs that do all fit in the L2, where $B$ is the bandwidth of the cache")
    else:
        layout_text = (rf"{layout_above} of the {pairs} measured pairs exceed the bound of their own layout, Eq.~\eqref{{eq:slayout}}, "
                       rf"{layout_above_resident} of them on matrices that fit in the L2")
    if above == 0:
        any_text = r"No speedup exceeds Eq.~\eqref{eq:sbound}"
    elif above == 1:
        any_text = r"Only one exceeds Eq.~\eqref{eq:sbound}" + (", again on a matrix in the L2" if above_resident else "")
    else:
        any_text = rf"Only {above} exceed Eq.~\eqref{{eq:sbound}}" + (", again all in the L2" if above == above_resident else "")
    return layout_text, any_text, capture, beyond_ceiling


def bound_brief_tex(records):
    layout_text, any_text, capture, _ = bound_check(records)
    return rf"""
The bounds of Sec.~\ref{{sec:model-sbound}} take one timing of CSR-i32 per matrix and precision.
{layout_text}.
{any_text}.
CSR-r16 reaches a median ${capture['fp64']:.2f}$ of its own bound at FP64, ${capture['fp32']:.2f}$ at FP32 and ${capture['fp16']:.2f}$ at FP16.
"""


def bound_tex(records):
    layout_text, any_text, capture, beyond_ceiling = bound_check(records)
    if not beyond_ceiling:
        ceiling_text = r"No measured speedup exceeds even $r_{\max}$"
    elif len(beyond_ceiling) == 1:
        record, label = beyond_ceiling[0]
        where = "a matrix that fits in the L2" if record["resident"] else "a matrix served from the DRAM"
        ceiling_text = (rf"Apart from {label} on \texttt{{{shared.tex_escape(record['matrix'])}}}, {where}, "
                        r"no measured speedup exceeds even $r_{\max}$")
    else:
        ceiling_text = rf"Only {len(beyond_ceiling)} measured speedups exceed even $r_{{\max}}$"
    lowest = min(record["efficiency"] for record in records)
    slowest = sorted({record["matrix"] for record in records if record["efficiency"] < 0.05})
    slowest_text = ", ".join(r"\texttt{" + shared.tex_escape(name) + "}" for name in slowest)
    return rf"""
\subsection{{The bound on the speedup}}
\label{{sec:results-bound}}

Both bounds cost the dimensions of the matrix and one timing of CSR-i32 on it.
{layout_text}.
{any_text}.
CSR-r16 reaches a median ${capture['fp64']:.2f}$ of its own bound at FP64, ${capture['fp32']:.2f}$ at FP32 and ${capture['fp16']:.2f}$ at FP16.
Eq.~\eqref{{eq:sbound}} is looser, since it covers an index of zero bytes that no layout approaches.
{ceiling_text}, including on matrices whose reference attains as little as ${lowest:.3f}$ of the roof ({slowest_text}, all with a few rows far longer than the rest)\extonly{{ (Fig.~\ref{{fig:bound}})}}.
The room the factor $B/K_{{\mathrm{{ref}}}}$ leaves above $r_{{\max}}$ is thus not used on any matrix served from the DRAM, which is an observation on this kernel and not a consequence of Eq.~\eqref{{eq:kleb}}.

\begin{{extended}}
\begin{{figure}}[t]
  \centering
  \includegraphics[width=\columnwidth]{{figures/bound.pdf}}
  \caption{{Measured speedup over CSR-i32 divided by $r_{{\max}}$, against the fraction of the DRAM roof the reference attains, one point per matrix, layout and precision.
  The curve is the bound of Eq.~\eqref{{eq:sbound}} and hollow markers are matrices that fit in the L2.}}
  \label{{fig:bound}}
\end{{figure}}
\end{{extended}}
"""


def implementation_tex(layout_gain, bound_gain):
    if layout_gain is None:
        return ""
    medians, size_change = layout_gain
    bound_text = ""
    brief_bound = ""
    brief_both = "it"
    if bound_gain is not None:
        brief_bound = (rf", while holding it to $32$ registers, at which it keeps as many warps resident as the other layouts, "
                       rf"adds ${bound_gain[0]['fp64']:.3f}\times$, ${bound_gain[0]['fp32']:.3f}\times$ and ${bound_gain[0]['fp16']:.3f}\times$")
        brief_both = "both"
        bounded = bound_gain[0]
        bound_text = (rf"Left to the compiler the lane-major kernel then took $40$ registers per thread against the $32$ of the other layouts, which on the A100 holds it at three quarters of their resident warps, and bounding it to $32$ without spilling adds a median ${bounded['fp64']:.3f}\times$ at FP64, ${bounded['fp32']:.3f}\times$ at FP32 and ${bounded['fp16']:.3f}\times$ at FP16." + "\n")
    return rf"""
\begin{{extended}}
\subsection{{The cost of an implementation}}
\label{{sec:results-implementation}}

The efficiency of a layout is a property of its implementation as much as of its encoding.
CSR-lp first stored the packed differences one entry at a time, so each lane rebuilt a 64-bit window from two global loads per nonzero, four in total with the value and the element of $x$.
Storing the same bits one lane at a time lets a lane keep the window in a register and load a new word only after consuming 32 bits, with the packed size within ${100.0 * size_change:.1f}$ per cent of the original.
That change alone makes CSR-lp faster by a median ${medians['fp64']:.2f}\times$ at FP64, ${medians['fp32']:.2f}\times$ at FP32 and ${medians['fp16']:.2f}\times$ at FP16, more than CSR-r16 gains over CSR-i32 at FP64.
{bound_text}Neither change alters the bytes $Q$ or the results, which stay bit-identical to CSR-i32, and every measurement of the preceding subsections includes both.
\end{{extended}}
"""


def intro_tex():
    return r"""% Generated by analysis/make_results.py; do not edit by hand.
\section{Results}
\label{sec:results}

This section reports how much each layout gains over CSR-i32\briefonly{ and}\extonly{,} what the rate ratio of Eq.~\eqref{eq:identity} consists of\extonly{, and what the implementation of a layout costs}.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "results" / "a100" / "csr_layouts_rcm.csv")
    parser.add_argument("--original-csv", type=Path, default=None)
    parser.add_argument("--element-major-csv", type=Path, default=None,
                        help="campaign with the element-major packing of CSR-lp")
    parser.add_argument("--unbounded-csv", type=Path, default=None,
                        help="campaign with the lane-major packing and no register bound")
    parser.add_argument("--tpr-csv", type=Path, default=None,
                        help="time of every layout at every threads per row (default <csv stem>_tpr.csv)")
    parser.add_argument("--ordering-log", type=Path, default=None,
                        help="campaign log with the time of every permutation (default <csv>.log)")
    parser.add_argument("--device", type=Path, default=None)
    parser.add_argument("--characterization", type=Path, default=None)
    parser.add_argument("--rate-study", type=Path, default=ROOT / "data" / "results" / "a100" / "k_study_rcm",
                        help="Nsight Compute exports of experiments/slurm/k_study.sbatch")
    parser.add_argument("--rate-list", type=Path, default=ROOT / "experiments" / "lists" / "k_study.txt")
    parser.add_argument("--rate-ordering", choices=("rcm", "original"), default="rcm",
                        help="ordering of the profiled runs: rcm for k_study_rcm, original for the earlier k_study")
    parser.add_argument("--rate-csv", type=Path, default=None,
                        help="campaign in the ordering the profile used, for the counted bytes and the T of "
                             "CSR-i32 (default --csv for rcm, csr_layouts.csv next to it for original)")
    parser.add_argument("--cusparse-csv", type=Path, default=None,
                        help="a later run of CSR-i32 and cuSPARSE with an FP32 compute type at FP16 "
                             "(experiments of the cuSPARSE fix), whose FP16 rows complete the campaign")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument("--fp16-controls", type=Path, default=ROOT / "data" / "results" / "a100" / "fp16_controls",
                        help="FP16 campaign repeated with the default, FP16-accumulate and unroll-16 builds")
    parser.add_argument("--cusparse-profile", type=Path, default=ROOT / "data" / "results" / "a100" / "cusparse_study",
                        help="Nsight Compute export of cuSPARSE beside CSR-i32, one <matrix>.ncu.csv per matrix")
    parser.add_argument("--second-device", type=Path, default=ROOT / "data" / "results" / "quadro",
                        help="the same reordered campaign on a second GPU, with its device.txt and _tpr.csv")
    parser.add_argument("--second-profile", type=Path, default=ROOT / "data" / "results" / "quadro" / "k_study",
                        help="Nsight Compute exports of the second GPU, one <matrix>.ncu.csv per matrix")
    parser.add_argument("--reps", type=int, default=20, help="timing samples of the campaign, for the prose")
    parser.add_argument("--warmup", type=int, default=3, help="warm-up products of the campaign, for the prose")
    parser.add_argument("--min-sample-ms", type=float, default=10.0,
                        help="minimum duration of a timing sample in the campaign, for the prose")
    args = parser.parse_args()
    original_csv = args.original_csv or args.csv.with_name("csr_layouts.csv")
    element_major_csv = args.element_major_csv or args.csv.parent / "previous" / args.csv.name

    device = load_device(args.csv, args.device)
    # The paper takes B to be the theoretical bandwidth of the device, which needs no benchmark to describe.
    device.read_bw_gbs = device.theoretical_bw_gbs
    shared.DEVICE = device
    own_table = shared.load_measurements(args.csv)
    separate_fp16 = add_cusparse_fp16(own_table, args.cusparse_csv) if args.cusparse_csv else False
    tpr_csv = args.tpr_csv or args.csv.with_name(args.csv.stem + "_tpr.csv")
    table, own_gain = at_reference_threads(own_table, load_thread_samples(tpr_csv))
    unbounded = shared.load_measurements(args.unbounded_csv) if args.unbounded_csv else None
    characterization = shared.load_characterization(args.characterization)
    records = summarize(table, device, characterization)
    identical, cusparse_deviation = shared.measurement_checks(table)
    identical_everywhere = all(float(row["max_abs_diff"]) == 0.0 for row in shared.read_rows(tpr_csv))
    for precision in identical:
        identical[precision] = identical[precision] + [identical_everywhere]
    cost = shared.ordering_seconds(args.ordering_log or args.csv.with_suffix(".log"))

    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    bound_figure(records, figures_dir)
    speedup_figure(records, figures_dir)
    band_figure(records, figures_dir)
    performance_figure(table, device, figures_dir)
    excluded = above_the_roof(table, device)
    roofline_table = {key: entries for key, entries in table.items() if key[0] not in excluded}
    roofline_matrices = {name: row for name, row in characterization.items() if name not in excluded}
    shared.roofline_panels(shared.summarize_fp64(roofline_table, roofline_matrices), roofline_table,
                           roofline_matrices, figures_dir)
    print(f"Roofline figure leaves out {len(excluded)} matrices above the DRAM roof: {', '.join(sorted(excluded))}")

    sections_dir = args.output_dir / "sections"
    sections_dir.mkdir(parents=True, exist_ok=True)
    measurements = (rate_measurements(args.rate_study, args.rate_list,
                                      args.rate_csv or (args.csv if args.rate_ordering == "rcm"
                                                        else args.csv.with_name("csr_layouts.csv")))
                    if args.rate_study.is_dir() else [])
    campaign, limits_rest = limits_tex(records, figures_dir, measurements)
    controls = fp16_controls(args.fp16_controls, device, characterization)
    vendor_profile = cusparse_profile(args.cusparse_profile, characterization)
    second = second_device(args.second_device, characterization, args.second_profile, args.rate_list)
    (sections_dir / "results.tex").write_text(
        intro_tex()
        + setup_tex(device, records, identical, cusparse_deviation, (args.reps, args.warmup, args.min_sample_ms),
                    len(admissible_original(original_csv)), cost, own_gain, separate_fp16)
        + "\n\\begin{comment}\n" + admissibility_tex(records, characterization) + "\\end{comment}\n"
        + gain_tex(records, vendor_profile)
        + performance_tex(table, device)
        + rate_tex(measurements, records, args.rate_ordering, campaign, controls, second)
        + limits_rest
        + "\n\\begin{comment}\n" + bound_tex(records) + "\\end{comment}\n"
        + implementation_tex(lane_packed_gain(element_major_csv, unbounded or own_table),
                             lane_packed_gain(args.unbounded_csv, own_table) if unbounded else None))
    print(f"device {device.summary()}; {len(of_precision(records, 'fp64'))} matrices; "
          f"bit-identical: { {precision: all(flags) for precision, flags in identical.items()} }")


if __name__ == "__main__":
    main()
