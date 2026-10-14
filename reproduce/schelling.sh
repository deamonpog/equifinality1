#!/usr/bin/env bash
# Case study 1 (Schelling segregation): rule library -> simulation -> training
# -> conformal sets.  Produces Table 1 and the numbers of Sect. 4.
#
#   bash reproduce/schelling.sh            full experiment (HPC scale)
#   bash reproduce/schelling.sh --smoke    tiny end-to-end check (minutes, CPU)
#
# Full-scale cost, for planning: building the library fingerprints 10,000
# candidate trees (about 16 CPU-hours); simulating 2,089 rules x 100 runs is
# about 8 GB of JSON and is embarrassingly parallel (see --n_tasks/--task_id
# in scripts/generate_ruletrees.py); training each model takes 30 epochs on a
# single 48-80 GB GPU.
#
# Set WORKERS to the number of CPU cores to use, and PYTHON to the interpreter.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
WORKERS="${WORKERS:-4}"

if [[ "${1:-}" == "--smoke" ]]; then
    "$PYTHON" scripts/generate_ruletrees.py --n_candidates 100 --max_depth 2 \
        --num_runs 5 --max_steps 200 --snapshot_ticks 50,100,150,200 \
        --output_dir GENERATED_DATA_smoke --seed 42 --workers "$WORKERS"
    "$PYTHON" scripts/train.py --config configs/ruletree_smoke.yaml
    "$PYTHON" scripts/train.py --config configs/ruletree_rowcol_smoke.yaml
    exit 0
fi

LIB=GENERATED_DATA/rule_trees/tree_library.json
mkdir -p GENERATED_DATA/rule_trees

# 1. The rule library. The 2,089 behaviorally distinct rules used in the paper
#    ship in data/schelling/tree_library.json. Set REBUILD_LIBRARY=1 to
#    regenerate it from the 10,000 candidates instead (same seed).
if [[ "${REBUILD_LIBRARY:-0}" == "1" ]]; then
    "$PYTHON" scripts/generate_ruletrees.py --library_only \
        --n_candidates 10000 --max_depth 3 --method mixed \
        --grid_size 50 --max_steps 500 --output_dir GENERATED_DATA \
        --seed 42 --workers "$WORKERS"
else
    cp data/schelling/tree_library.json "$LIB"
fi

# 2. Simulate every rule 100 times (50x50 torus, density 0.95, 500 ticks,
#    snapshots at ticks 100..500). On a cluster, split this with
#    --n_tasks N --task_id I (one array task per slice).
"$PYTHON" scripts/generate_ruletrees.py --simulate_only \
    --num_runs 100 --grid_size 50 --max_steps 500 \
    --snapshot_ticks 100,200,300,400,500 \
    --output_dir GENERATED_DATA --seed 42 --workers "$WORKERS"

# 3. Train the two models of Table 1: per-position and row+column embeddings.
"$PYTHON" scripts/train.py --config configs/ruletree_base.yaml
"$PYTHON" scripts/train.py --config configs/ruletree_rowcol.yaml

# 4. Conformal sets (Table 1). Calibrates RAPS on half of the held-out test
#    partition and evaluates on the other half; writes
#    results/logs/<name>/conformal_recalibrated.json.
"$PYTHON" scripts/recalibrate_conformal.py --config configs/ruletree_base.yaml
"$PYTHON" scripts/recalibrate_conformal.py --config configs/ruletree_rowcol.yaml
