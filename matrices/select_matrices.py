#!/usr/bin/env python3
"""select_matrices.py — reproducible matrix selection from the SuiteSparse index.

Reads ssstats.csv (downloaded if absent), maps the SuiteSparse `kind` field onto
application categories, keeps square real matrices, and writes manifest.csv.

Selection rule, per category and size tier:
  - tiers by nnz: S [1M,10M), M [10M,50M), L [50M,200M), XL [200M,500M]
    (500M keeps every matrix, with all benchmark layouts, inside a 16 GB GPU)
  - at most PER_TIER matrices (GRAPH_PER_TIER for the graph category)
  - structural twins (same group, rows and nnz within TWIN_TOLERANCE) are reduced
    to one, so near-identical siblings never occupy two slots
  - at most MAX_PER_GROUP matrices per SuiteSparse group within a tier, so that one
    family of near-identical matrices cannot dominate a category
  - among the eligible matrices, picks are spread evenly over the nnz range

The rule is deterministic: the same index always yields the same manifest.
"""
import argparse
import csv
import os
import urllib.request
from collections import defaultdict

INDEX_URL = "https://sparse.tamu.edu/files/ssstats.csv"
MATRIX_URL = "https://sparse.tamu.edu/MM/{group}/{name}.tar.gz"

CATEGORY_OF_KIND = {
    "structural problem": "structural",
    "subsequent structural problem": "structural",
    "structural problem sequence": "structural",
    "computational fluid dynamics problem": "cfd",
    "2D/3D problem": "pde",
    "model reduction problem": "pde",
    "semiconductor device problem": "pde",
    "semiconductor process problem": "pde",
    "materials problem": "pde",
    "electromagnetics problem": "electromagnetics",
    "circuit simulation problem": "circuit_power",
    "circuit simulation matrix": "circuit_power",
    "power network problem": "circuit_power",
    "optimization problem": "optimization",
    "theoretical/quantum chemistry problem": "chemistry",
    "undirected graph": "graph",
    "directed graph": "graph",
    "undirected weighted graph": "graph",
    "directed weighted graph": "graph",
    "undirected multigraph": "graph",
    "undirected random graph": "graph",
    "undirected graph with communities": "graph",
    "directed graph with communities": "graph",
    "random undirected weighted graph": "graph",
    "directed temporal multigraph": "graph",
}

TIERS = [("S", 1_000_000, 10_000_000), ("M", 10_000_000, 50_000_000),
         ("L", 50_000_000, 200_000_000), ("XL", 200_000_000, 500_000_001)]
PER_TIER = 4
GRAPH_PER_TIER = 3
MAX_PER_GROUP = 2
TWIN_TOLERANCE = 0.01


def load_index(path):
    if not os.path.exists(path):
        urllib.request.urlretrieve(INDEX_URL, path)
    with open(path) as index_file:
        lines = index_file.read().splitlines()[2:]      # line 1: count, line 2: date
    entries = []
    for group, name, rows, cols, nnz, is_real, is_binary, _is_nd, _posdef, \
            pattern_symmetry, numeric_symmetry, kind, _pattern_entries in csv.reader(lines):
        entries.append(dict(group=group, name=name, rows=int(rows), cols=int(cols), nnz=int(nnz),
                            is_real=int(is_real), is_binary=int(is_binary), kind=kind,
                            pattern_symmetry=float(pattern_symmetry),
                            numeric_symmetry=float(numeric_symmetry)))
    return entries


def tier_of(nnz):
    for tier_name, lower, upper in TIERS:
        if lower <= nnz < upper:
            return tier_name
    return None


def eligible(entry, category):
    if entry["rows"] != entry["cols"] or not entry["is_real"]:
        return False
    # Scientific categories need numerical values; graphs may be pattern-only.
    return category == "graph" or not entry["is_binary"]


def drop_structural_twins(candidates):
    """Keeps the first of any set of same-group matrices whose rows and nnz agree
    within TWIN_TOLERANCE (e.g. atmosmodl/atmosmodm, af_1_k101/af_shell3)."""
    kept = []
    for entry in sorted(candidates, key=lambda candidate: candidate["nnz"]):
        is_twin = any(other["group"] == entry["group"]
                      and abs(other["rows"] - entry["rows"]) <= TWIN_TOLERANCE * entry["rows"]
                      and abs(other["nnz"] - entry["nnz"]) <= TWIN_TOLERANCE * entry["nnz"]
                      for other in kept)
        if not is_twin:
            kept.append(entry)
    return kept


def spread_pick(candidates, count):
    """Pick `count` candidates evenly spaced over the nnz-sorted list, honouring
    the per-group cap."""
    ordered = sorted(candidates, key=lambda entry: entry["nnz"])
    picked, per_group = [], defaultdict(int)
    if not ordered:
        return picked
    step = max(1, len(ordered) // count)
    passes = [ordered[offset::step] for offset in range(step)]   # evenly spaced first, then fill gaps
    for sequence in passes:
        for entry in sequence:
            if len(picked) == count:
                return picked
            if entry in picked or per_group[entry["group"]] >= MAX_PER_GROUP:
                continue
            picked.append(entry)
            per_group[entry["group"]] += 1
    return picked


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--index", default=os.path.join(here, "ssstats.csv"))
    parser.add_argument("--manifest", default=os.path.join(here, "manifest.csv"))
    args = parser.parse_args()

    buckets = defaultdict(list)
    for entry in load_index(args.index):
        category = CATEGORY_OF_KIND.get(entry["kind"])
        tier = tier_of(entry["nnz"])
        if category and tier and eligible(entry, category):
            buckets[(category, tier)].append(entry)

    selected = []
    for (category, tier), candidates in sorted(buckets.items()):
        limit = GRAPH_PER_TIER if category == "graph" else PER_TIER
        for entry in spread_pick(drop_structural_twins(candidates), limit):
            selected.append(dict(category=category, tier=tier, **entry,
                                 url=MATRIX_URL.format(group=entry["group"], name=entry["name"])))

    fields = ["category", "tier", "group", "name", "rows", "cols", "nnz", "kind",
              "is_binary", "pattern_symmetry", "numeric_symmetry", "url"]
    with open(args.manifest, "w", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)

    counts = defaultdict(lambda: defaultdict(int))
    for entry in selected:
        counts[entry["category"]][entry["tier"]] += 1
    print(f"{'category':<18}" + "".join(f"{tier:>5}" for tier, _, _ in TIERS) + f"{'total':>7}")
    for category in sorted(counts):
        row = counts[category]
        print(f"{category:<18}" + "".join(f"{row[tier]:>5}" for tier, _, _ in TIERS) + f"{sum(row.values()):>7}")
    print(f"\n{len(selected)} matrices, {sum(entry['nnz'] for entry in selected)/1e9:.2f}e9 total nnz -> {args.manifest}")


if __name__ == "__main__":
    main()
