#!/usr/bin/env python3
"""Compare a new campaign with the published one, matrix by matrix.

  analysis/compare_campaign.py results/local/csr_layouts_rcm.csv [data/results/a100/csr_layouts_rcm.csv]

Absolute times depend on the device, so the check is on what must not change: every result is
bit-identical to CSR-i32, and the bytes per nonzero match the published ones. The speedups over CSR-i32
are printed beside the published ones, each layout at its own fastest threads per row.
"""
import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAYOUTS = {("absolute", "2"): "i16", ("row-base", "2"): "r16", ("lane-packed", "0"): "lp"}


def read_campaign(path):
    """(matrix, precision, layout) to its row, for the kernels of this repository that ran."""
    campaign = {}
    for row in csv.DictReader(open(path)):
        if row["status"] != "ok" or row["kernel"] != "custom":
            continue
        layout = "i32" if (row["column_mode"], row["column_bytes"]) == ("absolute", "4") else \
            LAYOUTS.get((row["column_mode"], row["column_bytes"]))
        if layout is not None:
            campaign[(row["matrix"], row["prec"], layout)] = row
    return campaign


def main():
    new = read_campaign(sys.argv[1])
    published = read_campaign(sys.argv[2] if len(sys.argv) > 2 else ROOT / "data/results/a100/csr_layouts_rcm.csv")
    common = sorted(key for key in new if key in published)
    differing = [key for key in new if float(new[key]["max_abs_diff"]) != 0.0]
    # CSR-lp packs a different stream at each threads per row, so its bytes follow the T each run chose.
    bytes_moved = [key for key in common if key[2] != "lp"
                   and abs(float(new[key]["bytes_total"]) / float(published[key]["bytes_total"]) - 1.0) > 1e-9]
    print(f"{len(new)} rows measured, {len(common)} also in the published campaign")
    print(f"results that differ from CSR-i32: {len(differing)} (must be 0)")
    print(f"byte counts that differ from the published ones: {len(bytes_moved)} (must be 0)")
    for precision in ("fp64", "fp32", "fp16"):
        for layout in ("i16", "r16", "lp"):
            pairs = []
            for key in common:
                if key[1:] != (precision, layout) or (key[0], precision, "i32") not in new:
                    continue
                pairs.append((float(new[(key[0], precision, "i32")]["t_ms"]) / float(new[key]["t_ms"]),
                              float(published[(key[0], precision, "i32")]["t_ms"]) / float(published[key]["t_ms"])))
            if pairs:
                print(f"  {precision} CSR-{layout:3s} n={len(pairs):2d} median speedup here "
                      f"{statistics.median(pair[0] for pair in pairs):.3f}, published "
                      f"{statistics.median(pair[1] for pair in pairs):.3f}")
    sys.exit(1 if differing or bytes_moved else 0)


if __name__ == "__main__":
    main()
