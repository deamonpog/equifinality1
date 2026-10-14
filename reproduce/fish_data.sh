#!/usr/bin/env bash
# Case study 2 (collective motion): preprocessing and aggregate baselines.
#
# Needs the Lei et al. (2020) release (figshare, doi:10.6084/m9.figshare.11858379)
# unpacked so that FISH_DATA/Agents/*.dat and FISH_DATA/Robots/*.dat exist.
# CPU only. Steps 01-06 take about ten minutes and the
# random forests of step 07 about an hour on a few cores.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"

"$PYTHON" scripts/fish/01_inspect_agent_fish_data.py      # results/fish/inspection.json
"$PYTHON" scripts/fish/02_create_rule_lookup.py           # data/fish/rule_lookup.csv
"$PYTHON" scripts/fish/03_preprocess_agent_trajectories.py  # data/fish/runs.npz
"$PYTHON" scripts/fish/04_create_grouped_splits.py        # data/fish/splits.json
"$PYTHON" scripts/fish/05_create_trajectory_windows.py    # data/fish/windows_T*.npz
"$PYTHON" scripts/fish/06_create_summary_features.py      # data/fish/summary_T*.npz
"$PYTHON" scripts/fish/07_train_summary_baselines.py      # results/fish/baselines.json (random forests)

# Optional checks of the feature code and the model's permutation invariance.
"$PYTHON" scripts/fish/verify_augmentation.py
"$PYTHON" scripts/fish/verify_gpu_transform.py
"$PYTHON" scripts/fish/verify_permutation_invariance.py
