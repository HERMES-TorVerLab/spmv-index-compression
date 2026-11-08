#!/usr/bin/env python3
"""make_layout_figure.py — schematic of how CSR-r16 and CSR-lp encode the column indices of one row.

Usage (from the repository root):
  python3 analysis/make_layout_figure.py --output-dir ../multi-precision/overleaf

Writes figures/layouts.pdf. The row is made up, ten nonzeros chosen so that the packed width of
CSR-lp grows with the threads per row T, and every number in the figure follows from the encoder
of src/bench_csr_width.cu (lane_delta, delta_width_bits, lane_major_bit).
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

import make_i16_results as shared  # noqa: E402

COLUMNS = [40, 42, 45, 47, 50, 53, 56, 59, 62, 65]
LANE_COLOUR = [shared.SERIES_BLUE, shared.SERIES_ORANGE, shared.SERIES_AQUA, "#8a5cc2"]
CELL = 1.0
LABEL_X = -0.25


def lane_deltas(columns, threads_per_row):
    """Delta of every entry from the entry T positions back, or from the first column for the first T."""
    return [column - columns[0] if index < threads_per_row else column - columns[index - threads_per_row]
            for index, column in enumerate(columns)]


def width_bits(deltas):
    return max(1, max(deltas).bit_length())


def lane_major(deltas, threads_per_row):
    """(delta, lane) in the order the packed stream stores them, one lane after the other."""
    return [(deltas[index], lane) for lane in range(threads_per_row)
            for index in range(lane, len(deltas), threads_per_row)]


def draw_row(axis, y, cells, label, note, lanes=None):
    for index, value in enumerate(cells):
        colour = LANE_COLOUR[lanes[index]] if lanes is not None else shared.NEUTRAL_FILL
        axis.add_patch(Rectangle((index * CELL, y), CELL, 0.8, facecolor=colour, alpha=0.25 if lanes else 0.5,
                                 edgecolor=colour if lanes else shared.INK_SECONDARY, linewidth=0.8))
        axis.text(index * CELL + CELL / 2, y + 0.4, str(value), ha="center", va="center", fontsize=6.5,
                  color=shared.INK_PRIMARY)
    axis.text(LABEL_X, y + 0.4, label, ha="right", va="center", fontsize=7, color=shared.INK_PRIMARY)
    axis.text(len(cells) * CELL + 0.25, y + 0.4, note, ha="left", va="center", fontsize=6.5,
              color=shared.INK_SECONDARY)


def layouts_figure(output):
    count = len(COLUMNS)
    four = lane_deltas(COLUMNS, 4)
    two = lane_deltas(COLUMNS, 2)
    stream = lane_major(four, 4)
    four_bits, two_bits = width_bits(four), width_bits(two)
    figure, axis = plt.subplots(figsize=(3.4, 2.05))

    draw_row(axis, 5.2, COLUMNS, "column $c_k$", "", [index % 4 for index in range(count)])
    draw_row(axis, 4.1, [column - COLUMNS[0] for column in COLUMNS], "CSR-r16", "16 bit each")
    draw_row(axis, 2.6, four, "CSR-lp, $T=4$", f"$w={four_bits}$ bit", [index % 4 for index in range(count)])
    draw_row(axis, 1.6, [value for value, _ in stream], "in memory", f"{-(-count * four_bits // 32)} words",
             [lane for _, lane in stream])
    draw_row(axis, 0.3, two, "CSR-lp, $T=2$", f"$w={two_bits}$ bit, {-(-count * two_bits // 32)} word",
             [index % 2 for index in range(count)])

    # one delta of CSR-lp at T = 4 against the entry it is taken from
    axis.add_patch(FancyArrowPatch((6 * CELL + CELL / 2, 3.45), (2 * CELL + CELL / 2, 3.45),
                                   connectionstyle="arc3,rad=0.25", arrowstyle="-|>", mutation_scale=6,
                                   color=shared.INK_SECONDARY, linewidth=0.7))
    axis.text(4 * CELL + CELL / 2, 3.75, f"${COLUMNS[6]}-{COLUMNS[2]}$", ha="center", va="bottom", fontsize=6,
              color=shared.INK_SECONDARY)
    # the 32-bit word boundary of the lane-major stream
    boundary = 32 // four_bits
    axis.plot([boundary * CELL] * 2, [1.45, 2.55], color=shared.INK_PRIMARY, linewidth=1.0)
    axis.text(boundary * CELL, 1.4, "32-bit word", ha="center", va="top", fontsize=6, color=shared.INK_SECONDARY)

    axis.set_xlim(LABEL_X - 2.6, count * CELL + 2.6)
    axis.set_ylim(0.2, 6.1)
    axis.axis("off")
    figure.tight_layout(pad=0.1)
    figure.savefig(output, dpi=300)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    (args.output_dir / "figures").mkdir(parents=True, exist_ok=True)
    layouts_figure(args.output_dir / "figures" / "layouts.pdf")


if __name__ == "__main__":
    main()
