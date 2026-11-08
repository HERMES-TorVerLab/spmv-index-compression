#!/usr/bin/env python3
"""make_dataset_tex.py — dataset section draft and tables from matrices/manifest.csv + characterization{,_rcm}/*.csv.

Writes the matrix-set section of the paper (selection, size, the structural properties that
govern index compression, the effect of RCM reordering on admissibility, and the per-matrix
ceiling S_max) with a per-domain summary table and a names table to output/dataset.tex, and the per-matrix
table to output/dataset_matrices.tex. The latter is a longtable for a one-column supplement:
longtable does not work in the two-column IEEE layout.

The section describes the data set only; implementation details (scripts, file formats) are
deliberately left out. Every number in the text is computed from the characterization files.

Requires \\usepackage{booktabs,longtable} in the including document.
One sentence per line; parentheses instead of em-dashes.
"""
import argparse
import csv
import glob
import math
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATRICES = os.path.join(ROOT, "matrices")
OUTPUT = os.path.join(ROOT, "output")
CATEGORY_LABEL = {
    "structural": "Structural", "cfd": "CFD", "pde": "2D/3D PDE", "electromagnetics": "Electromagn.",
    "circuit_power": "Circuit/power", "optimization": "Optimization", "chemistry": "Quantum chem.",
    "graph": "Graph",
}
CATEGORY_PROSE = {
    "structural": "structural mechanics", "cfd": "CFD", "pde": "2D/3D PDE", "electromagnetics": "electromagnetics",
    "circuit_power": "circuit and power networks", "optimization": "optimization", "chemistry": "quantum chemistry",
    "graph": "graphs",
}
TIER_ORDER = {"S": 0, "M": 1, "L": 2, "XL": 3}
WINDOW_16 = 65536


def tex_escape(text):
    return text.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def number(value, digits=2):
    return "--" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.{digits}f}"


def compact(value):
    """One decimal below 100 unless the value is whole; no decimals from 100 on."""
    if math.isnan(value):
        return "--"
    if value >= 100 or value == int(value):
        return f"{value:.0f}"
    return f"{value:.1f}"


def tier_of(nonzeros):
    """Size tier of a matrix by the nonzeros it stores, a symmetric matrix counted in full since that is
    what the kernel reads. The manifest keeps the tier of the selection, which follows the SuiteSparse
    count of one triangle, and differs from this one on two matrices of the set."""
    millions_of_nonzeros = nonzeros / 1e6
    if millions_of_nonzeros < 10:
        return "S"
    if millions_of_nonzeros < 50:
        return "M"
    return "L" if millions_of_nonzeros < 200 else "XL"


def millions(count):
    return f"{count / 1e6:.1f}"


def read_characterization(path):
    with open(path) as characterization_file:
        row = next(csv.DictReader(characterization_file))
    return {key: (value if key in ("name", "ordering") else float(value)) for key, value in row.items()}


def load(manifest_path, characterization_dir, reordered_dir):
    with open(manifest_path) as manifest_file:
        manifest = {row["name"]: row for row in csv.DictReader(manifest_file)}
    merged = []
    for path in sorted(glob.glob(os.path.join(characterization_dir, "*.csv"))):
        row = read_characterization(path)
        if row["name"] not in manifest:
            continue
        reordered_path = os.path.join(reordered_dir, os.path.basename(path))
        reordered = read_characterization(reordered_path) if os.path.exists(reordered_path) else None
        row["rcm_row_span_max"] = reordered["row_span_max"] if reordered else float("nan")
        row["rcm_rows_fitting_16bit"] = reordered["rows_fitting_16bit"] if reordered else float("nan")
        row.update(category=manifest[row["name"]]["category"], tier=tier_of(int(row["nnz"])),
                   kind=manifest[row["name"]]["kind"], group=manifest[row["name"]]["group"])
        merged.append(row)
    return manifest, merged


def median(values):
    finite = [value for value in values if not math.isnan(value)]
    return statistics.median(finite) if finite else float("nan")


def span_bits(span):
    return float("nan") if math.isnan(span) else float(max(1, math.ceil(math.log2(span + 1))))


def fits_16(row, prefix=""):
    return row[prefix + "rows_fitting_16bit"] == 1.0


def imbalance(row):
    return row["row_length_max"] / row["row_length_mean"]


def summary_table(rows):
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Matrix set by domain and tier (S/M/L/XL), with medians of $\ell$, $\ell_{\max}/\ell$, the bits"
             r" of the 90th-percentile row span and $r_{\max}$ at FP64, and the matrices whose row spans fit 16 bits.}",
             r"\label{tab:dataset-summary}", r"\footnotesize", r"\setlength{\tabcolsep}{2pt}",
             r"\begin{tabular}{lcrrrrrr}", r"\toprule",
             r" & & & & & \multicolumn{2}{c}{$s<2^{16}$} & \\", r"\cmidrule(lr){6-7}",
             r"domain & tiers & $\ell$ & $\ell_{\max}/\ell$ & bits $s_{90}$ & orig. & RCM & $r_{\max}$ \\",
             r"\midrule"]
    for category in CATEGORY_LABEL:
        members = [row for row in rows if row["category"] == category]
        if not members:
            continue
        tiers = "/".join(str(sum(1 for row in members if row["tier"] == tier)) for tier in TIER_ORDER)
        lines.append(
            f"{CATEGORY_LABEL[category]} & {tiers}"
            f" & {compact(median([row['row_length_mean'] for row in members]))}"
            f" & {compact(median([imbalance(row) for row in members]))}"
            f" & {compact(median([span_bits(row['row_span_p90']) for row in members]))}"
            f" & {sum(1 for row in members if fits_16(row))}/{len(members)}"
            f" & {sum(1 for row in members if fits_16(row, 'rcm_'))}/{len(members)}"
            f" & {number(median([row['speedup_ceiling'] for row in members]))} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return lines


def names_table(rows):
    """Compact reference for reproducibility: names only, by domain (rows) and size tier (columns),
    ordered by nonzero count inside each cell. Full-width float for the two-column layout."""
    tier_width = 0.21
    category_width = 0.10
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{The matrices of the set, by domain and size tier, in increasing number of nonzeros.}",
             r"\label{tab:dataset-names}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             rf"\begin{{tabular}}{{@{{}}p{{{category_width}\textwidth}}"
             + "".join(rf"p{{{tier_width}\textwidth}}" for _ in TIER_ORDER) + r"@{}}",
             r"\toprule",
             r"domain & S ($1$--$10$~M nnz) & M ($10$--$50$~M) & L ($50$--$200$~M) & XL ($200$--$500$~M) \tabularnewline",
             r"\midrule"]
    for category in CATEGORY_LABEL:
        members = [row for row in rows if row["category"] == category]
        if not members:
            continue
        cells = []
        for tier in TIER_ORDER:
            names = [row["name"] for row in sorted(members, key=lambda row: row["nnz"]) if row["tier"] == tier]
            cells.append(r"\raggedright " + (", ".join(rf"\texttt{{{tex_escape(name)}}}" for name in names) if names else "--"))
        lines.append(rf"\raggedright {CATEGORY_LABEL[category]} & " + " & ".join(cells) + r" \tabularnewline")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return lines


def matrix_table(rows):
    lines = [r"\begin{small}", r"\begin{longtable}{llrrrrrrr}",
             r"\caption{Structural characterization of every matrix."
             r" Bits: $\lceil\log_2(s_{\max}+1)\rceil$ for the widest row span, in the original ordering and after RCM."
             r" $r_{\max}$: FP64 byte-ratio ceiling of Eq.~\eqref{eq:rmax}.}"
             r"\label{tab:dataset-matrices}\\", r"\toprule",
             r"matrix & tier & $m$ [M] & nnz [M] & $\ell$ & $\ell_{\max}/\ell$ & bits orig. & bits RCM & $r_{\max}$ \\",
             r"\midrule", r"\endhead"]
    for category in CATEGORY_LABEL:
        members = sorted((row for row in rows if row["category"] == category),
                         key=lambda row: (TIER_ORDER[row["tier"]], row["nnz"]))
        if not members:
            continue
        lines.append(rf"\multicolumn{{9}}{{l}}{{\emph{{{CATEGORY_LABEL[category]}}}}} \\")
        for row in members:
            lines.append(
                f"\\texttt{{{tex_escape(row['name'])}}} & {row['tier']} & {millions(row['rows'])} & {millions(row['nnz'])}"
                f" & {number(row['row_length_mean'], 1)} & {number(imbalance(row), 1)}"
                f" & {number(span_bits(row['row_span_max']), 0)} & {number(span_bits(row['rcm_row_span_max']), 0)}"
                f" & {number(row['speedup_ceiling'])} \\\\")
    lines += [r"\bottomrule", r"\end{longtable}", r"\end{small}"]
    return lines


def join_names(names):
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def prose(rows):
    count = len(rows)
    tier_counts = [sum(1 for row in rows if row["tier"] == tier) for tier in TIER_ORDER]
    nnz_values = [row["nnz"] for row in rows]
    row_lengths = [row["row_length_mean"] for row in rows]
    ceilings = [row["speedup_ceiling"] for row in rows]
    working_sets = [row["working_set_mib_csr_i32"] for row in rows]
    absolute_16 = [row for row in rows if row["cols"] <= WINDOW_16]
    absolute_16_small = sum(1 for row in absolute_16 if row["tier"] == "S")
    fitting_before = sum(1 for row in rows if fits_16(row))
    fitting_after = sum(1 for row in rows if fits_16(row, "rcm_"))
    category_imbalance = {category: median([imbalance(row) for row in rows if row["category"] == category])
                          for category in CATEGORY_LABEL if any(row["category"] == category for row in rows)}
    gains = sorted(((sum(1 for row in rows if row["category"] == category and fits_16(row, "rcm_"))
                     - sum(1 for row in rows if row["category"] == category and fits_16(row)), category)
                    for category in category_imbalance), reverse=True)
    largest_gains = [CATEGORY_PROSE[category] for gain, category in gains[:3] if gain > 0]
    most_balanced = min(category_imbalance, key=category_imbalance.get)
    least_balanced = max(category_imbalance, key=category_imbalance.get)
    return rf"""\section{{Matrix set}}
\label{{sec:dataset}}

We use {count} square, real-valued matrices from the SuiteSparse Matrix Collection, chosen by a deterministic rule so that the set can be regenerated.
They cover seven scientific-computing domains (structural mechanics, CFD, 2D/3D PDE, electromagnetics, circuit and power networks, optimization and quantum chemistry) and, as a contrast case with short and irregular rows, graphs.
Within each domain the matrices are grouped by nonzero count into four tiers, S ($1$--$10$~M), M ($10$--$50$~M), L ($50$--$200$~M) and XL ($200$--$500$~M), the upper limit being the largest working set that fits a $16$~GB device.
Each domain contributes up to four matrices per tier (three for graphs), spread over the nonzero range; near-identical matrices of the same group are kept once, while families that span several tiers (such as \texttt{{nlpkkt120}}, \texttt{{nlpkkt160}} and \texttt{{nlpkkt200}}) are kept to vary size at fixed structure.
The set has {tier_counts[0]}, {tier_counts[1]}, {tier_counts[2]} and {tier_counts[3]} matrices in tiers S, M, L and XL, between ${min(nnz_values) / 1e6:.1f}$ and ${max(nnz_values) / 1e6:.0f}$ million nonzeros (symmetric matrices stored in full), and mean row lengths between ${min(row_lengths):.1f}$ and ${max(row_lengths):.0f}$.
Table~\ref{{tab:dataset-names}} lists the matrices by domain and tier.
All four tiers enter the evaluation; with FP64 values and 32-bit indices their CSR working sets range from ${min(working_sets):.0f}$~MiB to ${max(working_sets) / 1024:.1f}$~GiB.

Three structural properties govern index compression, and Table~\ref{{tab:dataset-summary}} summarises them by domain.
The mean row length $\ell = \mathrm{{nnz}}/m$ determines over how many nonzeros the per-row traffic is amortised.
The row-length imbalance $\ell_{{\max}}/\ell$ measures how unevenly the work is spread over the rows; its median per domain ranges from ${category_imbalance[most_balanced]:.1f}$ ({CATEGORY_PROSE[most_balanced]}) to ${category_imbalance[least_balanced]:.0f}$ ({CATEGORY_PROSE[least_balanced]}).
The row span of row $r$,
%
\begin{{equation}}
  s_r \;=\; \max_{{k \in r}} c_k \;-\; \min_{{k \in r}} c_k ,
  \label{{eq:rowspan}}
\end{{equation}}
%
is the width of the column window that the row touches, where $c_k$ is the column of nonzero $k$.
A column index stored as a 16-bit offset from a per-row base is exact only if $s_r < 2^{{16}}$ in every row, whereas an absolute 16-bit index requires $n \le 2^{{16}}$, which holds for only {len(absolute_16)} matrices ({absolute_16_small} of them in tier S).

Row spans depend on the ordering of rows and columns, so every matrix is characterized both in its original ordering and after a symmetric reverse Cuthill--McKee permutation $P A P^{{\top}}$.
Reordering raises the number of matrices whose row spans all fit 16 bits from {fitting_before} to {fitting_after}, mostly in {join_names(largest_gains)}; no matrix fits 8 bits in either ordering.

Finally, the dimensions of a matrix fix a ceiling on what any compression of its column indices can deliver at a given precision.
With $v$ bytes per value, $p$ bytes per row-pointer entry and the per-row traffic amortised over the nonzeros, $a = [(m+1)\,p + (n+m)\,v]/\mathrm{{nnz}}$, removing the column-index stream altogether gives
%
\begin{{equation}}
  S_{{\max}} \;=\; \frac{{v + 4 + a}}{{v + a}} .
  \label{{eq:smax}}
\end{{equation}}
%
At FP64 with 32-bit row pointers ($v = 8$, $p = 4$) a square matrix has $a \approx 20/\ell$, so $S_{{\max}}$ depends on the matrix only through $\ell$ and not on its ordering: it approaches $1.5$ for long rows and is $1.37$ at $\ell = 7$.
Over the set it ranges from ${min(ceilings):.2f}$ to ${max(ceilings):.2f}$ (median ${statistics.median(ceilings):.2f}$).
The per-matrix characterization is available in the artifact repository.
% TODO: cite the artifact repository (URL or DOI) once it is public.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=os.path.join(MATRICES, "manifest.csv"))
    parser.add_argument("--characterization", default=os.path.join(MATRICES, "characterization"))
    parser.add_argument("--characterization-rcm", default=os.path.join(MATRICES, "characterization_rcm"))
    parser.add_argument("--output", default=os.path.join(OUTPUT, "dataset.tex"))
    parser.add_argument("--matrices-output", default=os.path.join(OUTPUT, "dataset_matrices.tex"))
    args = parser.parse_args()

    manifest, rows = load(args.manifest, args.characterization, args.characterization_rcm)
    if len(rows) != len(manifest):
        print(f"warning: {len(rows)} of {len(manifest)} selected matrices are characterized")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.matrices_output)), exist_ok=True)
    body = prose(rows) + "\n" + "\n".join(summary_table(rows)) + "\n\n" + "\n".join(names_table(rows)) + "\n"
    with open(args.matrices_output, "w") as supplement:
        supplement.write("% One-column supplement; requires booktabs and longtable.\n"
                         "% Numbered S1 so the paper can cite it without a cross-document reference.\n"
                         "\\renewcommand{\\thetable}{S\\arabic{table}}\n"
                         + "\n".join(matrix_table(rows)) + "\n")
    with open(args.output, "w") as output:
        output.write(body)
    print(f"wrote {args.output} and {args.matrices_output}: {len(rows)} matrices")


if __name__ == "__main__":
    main()
