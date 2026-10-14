"""Build the window index and the per-fold feature scalers.

For each window length T and each fold, enumerate the windows of every
partition and fit a training-only robust scaler. Writes one file per window
length:

    data/fish/windows_T{T}.npz
        fold{k}_{split}    [N, 2] int64, (run_index, start_event)
        fold{k}_median     [F] float32, fitted on the fold's TRAIN runs only
        fold{k}_iqr        [F] float32
        T, stride_train, stride_eval

Labels are not stored: a window inherits rule_id / family_id /
neighbor_count / exp_id from its run, and runs.npz already holds those, so
duplicating them here would just create a second source of truth.

Run from the repository root:
    python scripts/fish/05_create_trajectory_windows.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

from genmod.fish.windows import CLIP, apply_scaler, fit_scaler, make_window_index

RUNS_PATH = os.path.join("data", "fish", "runs.npz")
SPLITS_PATH = os.path.join("data", "fish", "splits.json")
OUT_DIR = os.path.join("data", "fish")

# Ablation 1. T=128 is the primary setting; 256 and 512 are included because
# the baseline accuracy was still climbing steeply at 128, so the grid has to
# reach far enough to show where identifiability saturates. Runs hold ~1550
# events, so T=512 (~222 s) still yields 3 non-overlapping eval windows/run.
WINDOW_LENGTHS = [16, 32, 64, 128, 256, 512]
TRAIN_STRIDE_DIVISOR = 4            # train stride = T / 4, a 75% overlap


def main():
    data = np.load(RUNS_PATH, allow_pickle=True)
    feats = data["feats"]
    offsets = data["offsets"]
    exp_ids = data["exp_id"]
    rule_ids = data["rule_id"]
    feature_names = [str(s) for s in data["feature_names"]]

    with open(SPLITS_PATH) as fh:
        splits = json.load(fh)
    folds = splits["folds"]

    print("Feature tensor %s, %d runs, %d folds"
          % (feats.shape, len(offsets) - 1, len(folds)))

    for T in WINDOW_LENGTHS:
        stride_train = max(1, T // TRAIN_STRIDE_DIVISOR)
        stride_eval = T  # non-overlapping: required for valid conformal sets
        payload = {"T": np.int64(T),
                   "stride_train": np.int64(stride_train),
                   "stride_eval": np.int64(stride_eval),
                   "clip": np.float32(CLIP)}

        print("\n=== T = %d kicks   (train stride %d, eval stride %d)"
              % (T, stride_train, stride_eval))
        print("    %-5s %10s %8s %8s %8s" % ("fold", "train", "val", "calib", "test"))

        for f in folds:
            k = f["fold"]
            counts = {}
            for split in ("train", "val", "calib", "test"):
                stride = stride_train if split == "train" else stride_eval
                idx = make_window_index(offsets, exp_ids, f[split], T, stride)
                payload["fold%d_%s" % (k, split)] = idx
                counts[split] = idx.shape[0]

            train_runs = np.where(np.isin(exp_ids, f["train"]))[0]
            median, iqr = fit_scaler(feats, offsets, train_runs)
            payload["fold%d_median" % k] = median
            payload["fold%d_iqr" % k] = iqr

            print("    %-5d %10d %8d %8d %8d"
                  % (k, counts["train"], counts["val"], counts["calib"],
                     counts["test"]))

        out = os.path.join(OUT_DIR, "windows_T%d.npz" % T)
        np.savez_compressed(out, **payload)
        print("    wrote %s" % out)

    # ---- checks, on the primary setting -------------------------------
    T = 128
    payload = np.load(os.path.join(OUT_DIR, "windows_T%d.npz" % T))
    f0 = folds[0]

    print("\n[1] Leakage: does any run appear in two partitions of fold 0?")
    seen = {}
    clash = False
    for split in ("train", "val", "calib", "test"):
        for run_idx, _ in payload["fold0_%s" % split]:
            prev = seen.setdefault(int(run_idx), split)
            if prev != split:
                clash = True
    print("    a run's windows all live in one partition ... %s" % (not clash))
    assert not clash

    print("\n[2] Exchangeability: are calibration/test windows non-overlapping?")
    for split in ("calib", "test"):
        idx = payload["fold0_%s" % split]
        bad = 0
        for run_idx in np.unique(idx[:, 0]):
            starts = np.sort(idx[idx[:, 0] == run_idx, 1])
            if np.any(np.diff(starts) < T):
                bad += 1
        print("    %-6s runs with overlapping windows ... %d" % (split, bad))
        assert bad == 0

    print("\n[3] Class balance of the fold-0 windows")
    for split in ("train", "val", "calib", "test"):
        idx = payload["fold0_%s" % split]
        per_rule = np.bincount(rule_ids[idx[:, 0]], minlength=11)
        print("    %-6s %6d windows   per-rule min %4d max %4d"
              % (split, idx.shape[0], per_rule.min(), per_rule.max()))

    print("\n[4] Does the RAPS quantile exist at each alpha?")
    n_cal = payload["fold0_calib"].shape[0]
    for alpha in (0.05, 0.10, 0.20):
        need = int(np.ceil(1.0 / alpha)) - 1
        print("    alpha %.2f: needs n_cal >= %3d, have %d ... %s"
              % (alpha, need, n_cal, "ok" if n_cal >= need else "TOO FEW"))

    print("\n[5] Effect of the robust scaler on the heavy-tailed features")
    median = payload["fold0_median"]
    iqr = payload["fold0_iqr"]
    raw = np.load(RUNS_PATH)["feats"]
    scaled = apply_scaler(raw[:200000], median, iqr)
    print("    %-16s %10s %10s | %10s %10s"
          % ("", "raw min", "raw max", "scaled min", "scaled max"))
    for name in ("ax", "ay", "speed", "dt", "d_nn1"):
        j = feature_names.index(name)
        print("    %-16s %10.2f %10.2f | %10.2f %10.2f"
              % (name, raw[:200000, :, j].min(), raw[:200000, :, j].max(),
                 scaled[..., j].min(), scaled[..., j].max()))
    print("    (clipping at +/- %.0f IQR keeps the 570-sigma acceleration"
          % CLIP)
    print("     artifacts from dominating the input scale)")


if __name__ == "__main__":
    main()
