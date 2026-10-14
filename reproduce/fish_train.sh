#!/usr/bin/env bash
# Case study 2 (collective motion): train every model in the paper and write
# its per-fold test/calibration probabilities to results/fish/<tag>/.
# Run reproduce/fish_data.sh first. Needs a CUDA GPU; the field transformer
# needs about 23 GB of GPU memory at batch 32.
#
# Every tag below matches a directory shipped in results/fish/, so running
# this script overwrites the shipped probabilities with your own.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"

# GRU over the 13 per-event group statistics, at every observation length
# (Table 3, Fig. 2). T = 64 used 40 epochs; the other lengths used 60.
"$PYTHON" scripts/fish/08_train_gru_baseline.py --window 64 --epochs 40
for T in 16 32 128 256 512; do
    "$PYTHON" scripts/fish/08_train_gru_baseline.py --window "$T" --epochs 60
done

# Transformer on rotation-invariant per-agent tokens, 1.8M parameters
# (Fig. 1, Fig. 3, Table 2); its checkpoints are reused on the robots.
"$PYTHON" scripts/fish/09_train_fish_transformer.py --config configs/fish_transformer_inv.yaml

# Capacity controls (Table 2).
"$PYTHON" scripts/fish/09_train_fish_transformer.py \
    --config configs/fish_transformer_inv_small.yaml --tag transformer_inv_T64_small_verify
"$PYTHON" scripts/fish/09_train_fish_transformer.py --config configs/fish_transformer_inv_depth.yaml

# Raw-coordinate transformer, undertrained (40 epochs) and converged (120
# epochs, folds 0 and 1) (Sect. 3.2).
"$PYTHON" scripts/fish/09_train_fish_transformer.py --config configs/fish_transformer.yaml
"$PYTHON" scripts/fish/09_train_fish_transformer.py --config configs/fish_transformer_raw120.yaml

# Field transformer on 32x32 density/momentum grids (Sect. 5.3). One job per
# fold, as in the paper's runs; the fold probabilities are then pooled under
# the tag field_T64.
mkdir -p configs/fish_ablations results/fish/field_T64
for K in 0 1 2 3 4; do
    CFG="configs/fish_ablations/field_T64_shared_f${K}.yaml"
    sed -e "s/^tag: .*/tag: field_T64_f${K}/" -e "s/^folds: .*/folds: [${K}]/" \
        configs/fish_field_T64_l40s.yaml > "$CFG"
    "$PYTHON" scripts/fish/15_train_fish_field.py --config "$CFG"
    cp "results/fish/field_T64_f${K}/probs_fold${K}.npz" results/fish/field_T64/
done
