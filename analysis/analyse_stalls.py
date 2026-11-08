#!/usr/bin/env python3
"""analyse_stalls.py - does CSR-lp miss the model because its decoding is a dependency chain?

The two-term model prices instructions by throughput, dividing them by the rate at which the device
can issue them. That is the right price for independent instructions. The bit-packed decoder derives
the position of one index from the width of the one before it, so its arithmetic is a chain, and a
chain is paid in latency instead.

The test is a correlation rather than a single number. Across the matrices of experiments/lists/
stall_study.txt the residual of CSR-lp at FP16 runs from 1.88 down to 1.09, and the share of warp
stalls waiting on a fixed-latency result has to follow that order while CSR-i32 and CSR-r16 stay
flat. This script lines the two up and reports the rank correlation.

Usage:  analysis/analyse_stalls.py --stalls data/results/a100/stalls
"""
import argparse
import csv
import io
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VALUE_PRECISION = {"double": "fp64", "float": "fp32", "__half": "fp16"}
KERNEL_LAYOUT = {"k_csr_cumulative": "absolute", "k_csr_row_relative": "r16",
                 "k_csr_lane_packed": "lp"}
INDEX_WIDTH = {"int": 4, "unsigned int": 4, "unsigned short": 2, "unsigned char": 1}
STALLS = {"wait": "smsp__warp_issue_stalled_wait_per_warp_active.pct",
          "short": "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
          "long": "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
          "icache": "smsp__warp_issue_stalled_no_instruction_per_warp_active.pct"}


def spearman(first, second):
    """Rank correlation, which is what a monotone prediction deserves. Ties take the mean rank."""
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        result = [0.0] * len(values)
        position = 0
        while position < len(order):
            end = position
            while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
                end += 1
            mean_rank = (position + end) / 2.0 + 1.0
            for index in range(position, end + 1):
                result[order[index]] = mean_rank
            position = end + 1
        return result
    a, b = ranks(first), ranks(second)
    n = len(a)
    if n < 3:
        return float("nan")
    mean_a, mean_b = statistics.mean(a), statistics.mean(b)
    numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    spread = (sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b)) ** 0.5
    return numerator / spread if spread else float("nan")


def classify(kernel_name):
    """(layout, precision, threads per row) from a demangled kernel name, or None if not ours."""
    family = next((key for key in KERNEL_LAYOUT if key in kernel_name), None)
    signature = re.search(r"<([^>]*)>", kernel_name)
    if family is None or signature is None:
        return None
    arguments = [part.strip() for part in signature.group(1).split(",")]
    precision = VALUE_PRECISION.get(arguments[0])
    if precision is None:
        return None
    layout = KERNEL_LAYOUT[family]
    if layout == "absolute":
        width = INDEX_WIDTH.get(arguments[1])
        if width is None:
            return None
        layout = "i" + str(width * 8)
    # the threads per row are a non-type template parameter, which one version of the tool prints as
    # "8" and another as "(int)8"
    digits = re.search(r"(\d+)\s*$", arguments[-1])
    if digits is None:
        return None
    return layout, precision, int(digits.group(1))


NUMERIC = re.compile(r"^[0-9]+(?:[.,][0-9]+)*$")


def decimal_separator(cells):
    """Which of . and , this file uses as the decimal point.

    Nsight Compute formats through the locale of the job, so the same metric arrives as "94.38" from
    one machine and "94,38" from another, and a count as "7785152", "7,785,152" or "7.785.152". The
    convention cannot be read off a single cell, since "1,234" is a thousand in one locale and one
    and a bit in the other, but it can be read off a file. Two facts settle it. When a cell carries
    both marks the rightmost one is the decimal point, and a grouping separator is always followed by
    exactly three digits, so any mark followed by a different number of digits is a decimal point.
    """
    for cell in cells:
        if not NUMERIC.match(cell):
            continue
        if "," in cell and "." in cell:
            return "," if cell.rindex(",") > cell.rindex(".") else "."
        for mark in (",", "."):
            if mark in cell and len(cell) - cell.rindex(mark) - 1 != 3:
                return mark
    return "."


def number(cell, decimal="."):
    """One number out of a Nsight Compute cell, given the decimal point the file uses."""
    cell = str(cell).strip()
    if not NUMERIC.match(cell):
        return None
    grouping = "." if decimal == "," else ","
    body = cell.replace(grouping, "")
    if body.count(decimal) > 1:          # a file that groups with the character it also decimates on
        head, _, tail = body.rpartition(decimal)
        body = head.replace(decimal, "") + decimal + tail
    try:
        return float(body.replace(decimal, "."))
    except ValueError:
        return None


# Prefixes of the units Nsight Compute scales a column to, and the base each lands on: times in
# nanoseconds, everything else (bytes, sectors, instructions, cycles) in plain counts.
TIME_SCALE = {"ns": 1.0, "nsecond": 1.0, "us": 1e3, "usecond": 1e3, "ms": 1e6, "msecond": 1e6,
              "s": 1e9, "second": 1e9}
PREFIX_SCALE = {"K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}


def unit_scale(unit):
    """The factor that brings a value printed in this unit to the base the analyses assume.

    A report exported with ncu --import is scaled for reading unless --print-units base is given, so
    a time arrives in microseconds or milliseconds and a count as Kinst or Mbyte, and the unit sits in
    the row under the header. The prefixes are decimal, as the tool prints them.
    """
    unit = (unit or "").strip()
    if unit in TIME_SCALE:
        return TIME_SCALE[unit]
    if len(unit) > 1 and unit[0] in PREFIX_SCALE and unit[1:] in ("byte", "inst", "sector", "cycle", "warp",
                                                                  "request", "thread"):
        return PREFIX_SCALE[unit[0]]
    return 1.0


def read_ncu(path):
    """Counters per profiled kernel, as {(layout, precision, threads): {metric: value}}.

    Four things about the file shape. A preamble precedes the table, the row under the header holds
    the units rather than data, a unit may carry a prefix that is undone here, and a configuration
    launched more than once appears more than once, so repeated launches are reduced by their median.
    """
    lines = path.read_text(errors="replace").splitlines()
    start = next((index for index, line in enumerate(lines)
                  if line.startswith('"ID"') or line.startswith("ID,")), None)
    if start is None:
        return {}
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
    # The convention is settled per column rather than per file, because the tool formats some
    # metrics when it collects them and others when it exports them, so one file can hold both. A
    # column is written by one of the two and is therefore consistent with itself. Exporting a
    # report with LC_ALL=C avoids the question, and a file collected elsewhere does not get to.
    metrics = [column for column in (rows[0] if rows else {}) if column and "__" in column]
    decimal = {column: decimal_separator(str(row[column]).strip() for row in rows)
               for column in metrics}
    units = rows[0] if rows and not (rows[0].get("Kernel Name") or rows[0].get("Demangled Name")) else {}
    scale = {column: unit_scale(units.get(column)) for column in metrics}
    gathered = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = classify(row.get("Kernel Name") or row.get("Demangled Name") or "")
        if key is None:                       # the units row and anything not ours land here
            continue
        for column in metrics:
            value = number(row[column], decimal[column])
            if value is not None:
                gathered[key][column.split("[")[0].strip()].append(value * scale[column])
    return {key: {metric: statistics.median(values) for metric, values in metrics.items()}
            for key, metrics in gathered.items()}


def pick(records, layout, precision, preferred):
    """The profiled instantiation that stands for a layout at a precision.

    The sweep over threads per row is not the same for every layout. CSR-i32 and CSR-r16 are profiled
    at all five candidates while CSR-lp is profiled only at the one the benchmark settled on, so
    insisting on the threads per row the campaign reported silently drops most of the CSR-lp rows.
    The campaign's choice is used when it was profiled, and the fastest profiled one otherwise.
    """
    candidates = {key[2]: value for key, value in records.items()
                  if key[0] == layout and key[1] == precision}
    if not candidates:
        return None, None
    if preferred in candidates:
        return candidates[preferred], preferred
    threads = min(candidates, key=lambda t: candidates[t].get("gpu__time_duration.sum", float("inf")))
    return candidates[threads], threads


def model_residuals(campaign, precision):
    """Measured time over the two-term model, per (matrix, layout), which is the quantity the stall
    shares have to explain. Rebuilt here from the same inputs the paper uses rather than read from a
    file, so the two numbers can never drift apart."""
    directory = campaign.parent
    device = {}
    for line in (directory / "device.txt").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            device[key.strip()] = value.strip()
    bandwidth = float(device["read_bw_gbs"])
    issue = float(device["sm"]) * float(device["clock_ghz"]) * 1e9 * 4.0 * 32.0
    work = {(row["layout"], row["precision"], int(row["threads_per_row"])): float(row["instructions_per_nnz"])
            for row in csv.DictReader(open(directory / "instructions.csv", newline=""))}
    occupancy = {}
    for name in ("characterization_v3", "characterization_v2", "characterization"):
        folder = ROOT / "matrices" / name
        if not folder.exists():
            continue
        for path in folder.glob("*.csv"):
            row = next(iter(csv.DictReader(open(path, newline=""))))
            occupancy.setdefault(row["name"], row)
    residual = {}
    for row in csv.DictReader(open(campaign, newline="")):
        if row["status"] != "ok" or row["prec"] != precision or row["kernel"] != "custom":
            continue
        if row["row_encoding"] != "cumulative":
            continue
        layout = {"lane-packed": "lp", "row-base": "r16"}.get(row["column_mode"])
        if layout is None and row["column_mode"] == "absolute":
            layout = "i" + str(int(row["column_bytes"]) * 8)
        if layout is None or row["matrix"] not in occupancy:
            continue
        threads = int(row["best_tpr"])
        lanes = float(occupancy[row["matrix"]].get(f"lane_occupancy_t{threads}", 1.0))
        modelled = (float(row["bytes_per_nnz"]) / bandwidth * 1e3
                    + work.get((layout, precision, threads), 0.0) / lanes / issue * 1e12)
        if modelled > 0:
            residual[(row["matrix"], layout)] = (2000.0 / float(row["gflops"])) / modelled
    return residual


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stalls", type=Path, default=ROOT / "data" / "results" / "a100" / "stalls")
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "results" / "a100" / "csr_layouts.csv")
    parser.add_argument("--precision", default="fp16")
    args = parser.parse_args()

    # the best threads per row of the campaign, so the counters are read off the configuration the
    # paper actually reports rather than off whichever instantiation ran first
    best = {}
    for row in csv.DictReader(open(args.csv, newline="")):
        if row["status"] != "ok" or row["row_encoding"] != "cumulative" or row["kernel"] != "custom":
            continue
        name = {"lane-packed": "lp", "row-base": "r16"}.get(row["column_mode"])
        if name is None and row["column_mode"] == "absolute":
            name = "i" + str(int(row["column_bytes"]) * 8)
        if name:
            best[(row["matrix"], row["prec"], name)] = int(row["best_tpr"])

    nonzeros = {}
    modelled = {}
    for row in csv.DictReader(open(args.csv, newline="")):
        if row["status"] != "ok" or row["row_encoding"] != "cumulative" or row["kernel"] != "custom":
            continue
        name = {"lane-packed": "lp", "row-base": "r16"}.get(row["column_mode"])
        if name is None and row["column_mode"] == "absolute":
            name = "i" + str(int(row["column_bytes"]) * 8)
        if name:
            nonzeros[row["matrix"]] = int(row["nnz"])
            modelled[(row["matrix"], row["prec"], name)] = float(row["bytes_per_nnz"])
    device = {}
    for line in (args.csv.parent / "device.txt").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            device[key.strip()] = value.strip()
    bandwidth = float(device["read_bw_gbs"])
    residuals = model_residuals(args.csv, args.precision)

    files = sorted(args.stalls.glob("*.ncu.csv"))
    if not files:
        raise SystemExit(f"{args.stalls}: no *.ncu.csv, run experiments/slurm/stall_study.sbatch first")

    print(f"{len(files)} matrices profiled at {args.precision}\n")

    rows = []
    for path in files:
        matrix = path.name[:-len(".ncu.csv")]
        records = read_ncu(path)
        for layout in ("i32", "r16", "lp"):
            preferred = best.get((matrix, args.precision, layout))
            entry, threads = pick(records, layout, args.precision, preferred)
            if entry is None or matrix not in nonzeros:
                continue
            moved = entry.get("dram__bytes_read.sum")
            elapsed = entry.get("gpu__time_duration.sum")
            rows.append({
                "matrix": matrix, "layout": layout, "threads": threads,
                "consistent": threads == preferred,
                "modelled_bytes": modelled.get((matrix, args.precision, layout)),
                "dram_bytes": moved / nonzeros[matrix] if moved else None,
                # bytes per nanosecond is gigabytes per second
                "bandwidth": moved / elapsed if moved and elapsed else None,
                "residual": residuals.get((matrix, layout)),
                "stalls": [entry.get(metric) for metric in STALLS.values()]})

    print("the numerator of the model: bytes per nonzero it counts, against what the DRAM moved")
    print(f"{'matrix':18s}{'layout':7s}{'model':>8s}{'DRAM':>8s}{'ratio':>8s}")
    for row in rows:
        if row["modelled_bytes"] and row["dram_bytes"]:
            print(f"{row['matrix']:18s}{row['layout']:7s}{row['modelled_bytes']:8.2f}"
                  f"{row['dram_bytes']:8.2f}{row['dram_bytes'] / row['modelled_bytes']:8.2f}")
    for layout in ("i32", "r16", "lp"):
        ratios = [row["dram_bytes"] / row["modelled_bytes"] for row in rows
                  if row["layout"] == layout and row["modelled_bytes"] and row["dram_bytes"]]
        if ratios:
            print(f"  median {layout:5s} {statistics.median(ratios):.2f}")

    print("\nthe denominator of the model: it assumes every layout attains the measured roof")
    print(f"{'matrix':18s}" + "".join(f"{layout:>18s}" for layout in ("i32", "r16", "lp")))
    print(f"{'':18s}" + "".join(f"{'GB/s':>10s}{'%roof':>8s}" for _ in range(3)))
    for matrix in sorted({row["matrix"] for row in rows}):
        cells = []
        for layout in ("i32", "r16", "lp"):
            row = next((r for r in rows if r["matrix"] == matrix and r["layout"] == layout), None)
            cells.append(f"{row['bandwidth']:10.0f}{100 * row['bandwidth'] / bandwidth:7.0f}%"
                         + ("" if row["consistent"] else "*") if row and row["bandwidth"]
                         else f"{'-':>18s}")
        print(f"{matrix:18s}" + "".join(cell.ljust(18) for cell in cells))
    print(f"{'median':18s}", end="")
    for layout in ("i32", "r16", "lp"):
        values = [row["bandwidth"] for row in rows if row["layout"] == layout and row["bandwidth"]]
        middle = statistics.median(values)
        print(f"{middle:10.0f}{100 * middle / bandwidth:7.0f}%".ljust(18), end="")
    print("\n* a threads per row the campaign did not choose, so the residual beside it is not of the"
          "\n  same configuration and the pair is left out of the correlation below")

    consistent = [row for row in rows if row["consistent"] and row["bandwidth"] and row["residual"]]
    print(f"\nrank correlation with the model residual, over {len(consistent)} consistent pairs")
    print(f"  attained bandwidth        rho = "
          f"{spearman([r['bandwidth'] for r in consistent], [r['residual'] for r in consistent]):+.2f}")
    waits = [r for r in consistent if r["stalls"][0] is not None]
    print(f"  stalled on a fixed-latency result   rho = "
          f"{spearman([r['stalls'][0] for r in waits], [r['residual'] for r in waits]):+.2f}")

    print("\nstall shares, as a percentage of the cycles a warp was active")
    print(f"{'layout':8s}" + "".join(f"{label:>9s}" for label in STALLS) + f"{'n':>5s}")
    for layout in ("i32", "r16", "lp"):
        selected = [row["stalls"] for row in rows if row["layout"] == layout and row["stalls"][0] is not None]
        if selected:
            print(f"{layout:8s}" + "".join(f"{statistics.median([s[index] for s in selected]):9.1f}"
                                           for index in range(len(STALLS))) + f"{len(selected):5d}")


if __name__ == "__main__":
    main()
