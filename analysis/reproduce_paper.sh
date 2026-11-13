#!/usr/bin/env bash
# Regenerate the Results section and its figures from the measurements of the data record.
#   analysis/reproduce_paper.sh [results directory, default data/results] [output directory, default output/paper]
set -euo pipefail
DATA="${1:-data/results}"
OUT="${2:-output/paper}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
A="${DATA}/a100"; Q="${DATA}/quadro"
python3 analysis/make_results.py \
  --csv "${A}/csr_layouts_rcm.csv" --ordering-log "${A}/csr_layouts_rcm.log" --device "${A}/device.txt" \
  --original-csv "${A}/csr_layouts.csv" --element-major-csv "${A}/element_major/csr_layouts_rcm.csv" \
  --unbounded-csv "${A}/unbounded/csr_layouts_rcm.csv" --rate-study "${A}/k_study_rcm" --rate-ordering rcm \
  --cusparse-csv "${A}/cusparse_rcm/cusparse_rcm.csv" --fp16-controls "${A}/fp16_controls" \
  --cusparse-profile "${A}/cusparse_study" --second-device "${Q}" --second-profile "${Q}/k_study" \
  --r16-loads "${A}/r16_loads" --second-r16-loads "${Q}/r16_loads" \
  --original-rate-study "${A}/k_study_original" --original-rate-csv "${A}/csr_layouts.csv" \
  --output-dir "${OUT}"
