"""Recompute the Schelling conformal results from a saved checkpoint.

No retraining. This loads results/checkpoints/<name>/best.pt, runs one forward
pass over the held-out data, and redoes the conformal step correctly.

Three defects are fixed relative to what scripts/train.py originally computed.

1. Missing randomization at rank 0.
   The APS score is  sum_{j ranked above y} p_j + u * p_y,  u ~ U(0,1).
   compute_nonconformity_scores applied the u term only when the true class was
   NOT ranked first, so for an accurate model most calibration points were
   assigned the full p_max instead of u * p_max. Their scores were inflated,
   q_hat was inflated with them, and every prediction set came out too large.

2. Deterministic prediction against randomized calibration.
   conformal_predict always included the class that crossed the threshold while
   calibration randomized it, so the two used different rules.

   Both are now fixed in genmod/evaluation/conformal.py.

3. The calibration set was drawn from the VALIDATION split.
   train.py took cal_probs from val_loader -- the same data used for early
   stopping and for fitting the temperature. Conformal requires the calibration
   scores to be exchangeable with the test scores, and a split that drove model
   SELECTION is not: the model has been tuned toward it, so its scores are
   optimistically low and the coverage guarantee does not hold as computed.

   Fixed here by carving the calibration set out of the TEST split, which was
   never seen during training or selection. The model's weights do not depend
   on which held-out data we calibrate on, so this needs no retraining -- only
   a re-partition and one inference pass.

The test partition is split in half: the first half calibrates, the second half
is evaluated. The split is made AFTER a fixed shuffle so it does not inherit
any ordering by class.

Usage:
    python scripts/recalibrate_conformal.py --config configs/ruletree_base.yaml
    python scripts/recalibrate_conformal.py --config configs/ruletree_rowcol.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")

from genmod.evaluation.conformal import run_conformal_at_multiple_alphas
from genmod.evaluation.metrics import collect_predictions
from genmod.models.factory import build_model
from genmod.utils.config import load_config
from scripts.train import SYSTEM_LOADERS

CAL_FRACTION = 0.5  # of the TEST split
SEED = 20260714


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None,
                    help="defaults to results/checkpoints/<cfg.name>/best.pt")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Config %s   name %s   device %s" % (args.config, cfg.name, device))

    # Rebuild exactly the model and splits that train.py built. The splits are
    # seeded off cfg.training.seed, so this reproduces the same partitions.
    loader_fn = SYSTEM_LOADERS[cfg.data.system]
    (train_ds, val_ds, test_ds, n_classes,
     label_mapping, tok, get_group, _) = loader_fn(cfg)
    print("Classes: %d   test samples: %d" % (n_classes, len(test_ds)))

    model = build_model(
        num_classes=n_classes,
        vocab_size=tok["vocab_size"],
        seq_len=tok["seq_len"],
        time_size=tok["time_size"],
        space_size=tok["space_size"],
        model_size=cfg.model.model_size,
        dropout=cfg.model.dropout,
        positional_encoding=cfg.model.positional_encoding,
        grid_rows=tok.get("grid_rows"),
        grid_cols=tok.get("grid_cols"),
    ).to(device)

    ckpt_path = Path(args.checkpoint or
                     Path(cfg.training.checkpoint_dir) / cfg.name / "best.pt")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()
    print("Loaded %s" % ckpt_path)

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=cfg.training.batch_size, shuffle=False)
    probs, _, labels = collect_predictions(model, test_loader, device)
    acc = float((probs.argmax(1) == labels).mean())
    print("Test top-1 accuracy (sanity check): %.4f" % acc)

    # Split the TEST partition into calibration and evaluation halves. Shuffle
    # first: the loader emits samples grouped by class, so a contiguous split
    # would put entire classes on one side and destroy exchangeability just as
    # surely as calibrating on validation did.
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(labels))
    n_cal = int(len(labels) * CAL_FRACTION)
    cal_idx, eval_idx = order[:n_cal], order[n_cal:]

    cal_probs, cal_labels = probs[cal_idx], labels[cal_idx]
    ev_probs, ev_labels = probs[eval_idx], labels[eval_idx]
    print("Calibration: %d samples   Evaluation: %d samples"
          % (len(cal_labels), len(ev_labels)))

    # With n_cal calibration points the RAPS quantile sits at
    # ceil((n+1)(1-alpha))/n, attainable only when n >= 1/alpha - 1.
    for alpha in cfg.conformal.alpha_levels:
        need = int(np.ceil(1.0 / alpha)) - 1
        ok = "ok" if len(cal_labels) >= need else "TOO FEW"
        print("  alpha %.2f needs n_cal >= %4d, have %d ... %s"
              % (alpha, need, len(cal_labels), ok))

    np.random.seed(SEED)  # the RAPS tie-breaking randomization
    group_labels = np.array([get_group(l) for l in ev_labels])

    results = run_conformal_at_multiple_alphas(
        cal_probs, cal_labels, ev_probs, ev_labels, group_labels,
        alpha_levels=cfg.conformal.alpha_levels,
        method=cfg.conformal.method,
        k_reg=cfg.conformal.k_reg,
        lambda_reg=cfg.conformal.lambda_reg,
    )

    print("\n%-7s %-10s %-10s %-12s %-10s" % (
        "alpha", "target", "coverage", "mean |C|", "singleton"))
    for alpha, r in sorted(results.items()):
        print("%-7.2f %-10.3f %-10.3f %-12.1f %-10.3f"
              % (alpha, 1 - alpha, r["coverage"], r["avg_set_size"],
                 r["singleton_fraction"]))

    out = Path(cfg.training.log_dir) / cfg.name / "conformal_recalibrated.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump({
            "note": ("Recomputed from best.pt with the fixed RAPS scoring and "
                     "prediction, and with the calibration set carved out of "
                     "TEST rather than VALIDATION."),
            "checkpoint": str(ckpt_path),
            "test_top1": acc,
            "n_calibration": int(len(cal_labels)),
            "n_evaluation": int(len(ev_labels)),
            "results": {str(k): v for k, v in results.items()},
        }, fh, indent=2, default=str)
    print("\nWrote %s" % out)


if __name__ == "__main__":
    main()
