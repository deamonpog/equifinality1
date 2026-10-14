#!/usr/bin/env bash
# Case study 2 and 3: everything downstream of the trained models.
#
# Works from the probabilities and checkpoints shipped in results/fish/, so it
# needs no retraining. Conformal sets, evaluation and capacity statistics need
# only results/; the capacity statistics, robot transfer and robot geometry
# also need data/fish/runs.npz and FISH_DATA/Robots (reproduce/fish_data.sh).
# The robot transfer retrains the GRU on each fold (about a minute per fold on
# a GPU, much longer on a CPU).
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"

TAGS="gru_T16 gru_T32 gru_T64 gru_T128 gru_T256 gru_T512 transformer_inv_T64
      transformer_inv_T64_small_verify transformer_inv_T64_depth field_T64
      transformer_T64"

# RAPS conformal sets: (k_reg, lambda) = (2, 0.1) -> results/fish/<tag>/conformal.json
for t in $TAGS; do "$PYTHON" scripts/fish/10_calibrate_fish_raps.py --tag "$t"; done

# Accuracy, family/k accuracy, calibration, run-level bootstrap CIs
# -> results/fish/<tag>/evaluation.json
for t in gru_T16 gru_T32 gru_T64 gru_T128 gru_T256 gru_T512 transformer_inv_T64 transformer_T64; do
    "$PYTHON" scripts/fish/11_evaluate_fish_rules.py --tag "$t"
done

# Table 2 and the paired GRU-transformer difference -> results/fish/capacity_stats.json
"$PYTHON" scripts/fish/12_capacity_stats.py

# Figures 1-3 of the paper (fig1_ladder_v6, fig6_equifinality_v6 = Fig. 2,
# fig4_confusion_v6 = Fig. 3) -> results/figures/fish/
"$PYTHON" scripts/fish/13_make_figures.py

# Robot transfer, transformer and GRU over five folds, time in each domain's
# median kick interval (Sect. 6, Tables 4-5) -> results/fish/robot_transfer_comparison.json
"$PYTHON" scripts/fish/15_gru_robot_transfer.py --window 64

# Sensitivity to the robot time unit (0.84, 0.92, 0.98 s) -> results/fish/robot_tau_sweep/
mkdir -p results/fish/robot_tau_sweep
for spec in legacy:0.840 pooled:0.920 filemean:0.980; do
    label="${spec%%:*}"; tau="${spec##*:}"
    "$PYTHON" scripts/fish/15_gru_robot_transfer.py --window 64 \
        --tau-robot "$tau" --tau-agent 0.434 --out "results/fish/robot_tau_sweep/15_${label}.json"
done

# Group geometry, Spearman correlation, paired t-tests -> results/fish/robot_geometry.json
"$PYTHON" scripts/fish/16_robot_geometry.py

# Compare everything with the numbers printed in the paper.
"$PYTHON" scripts/paper_numbers.py
