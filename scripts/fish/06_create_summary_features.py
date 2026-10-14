"""Build window-level group statistics for the aggregate baselines.

For each window length, collapse every window into a fixed-length vector of
group statistics (see genmod/fish/summary.py). These vectors have no notion
of which agent is which, so any model trained on them is testing how much of
the rule signal survives aggregation.

Writes data/fish/summary_T{T}.npz with one matrix per (fold, split), plus the
labels, so 07 can train without touching the raw feature tensor again.

Run from the repository root:
    python scripts/fish/06_create_summary_features.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

from genmod.fish.summary import SUMMARY_NAMES, summarize_window

RUNS_PATH = os.path.join("data", "fish", "runs.npz")
OUT_DIR = os.path.join("data", "fish")
WINDOW_LENGTHS = [16, 32, 64, 128, 256, 512]
SPLITS = ("train", "val", "calib", "test")


def main():
    data = np.load(RUNS_PATH, allow_pickle=True)
    feats = data["feats"]
    offsets = data["offsets"]
    feature_names = [str(s) for s in data["feature_names"]]
    rule_id = data["rule_id"]
    family_id = data["family_id"]
    neighbor_count = data["neighbor_count"]
    exp_id = data["exp_id"]

    print("Summary features: %d statistics per window" % len(SUMMARY_NAMES))

    for T in WINDOW_LENGTHS:
        win_path = os.path.join(OUT_DIR, "windows_T%d.npz" % T)
        wins = np.load(win_path)
        n_folds = len({k.split("_")[0] for k in wins.files if k.startswith("fold")})

        payload = {"summary_names": np.array(SUMMARY_NAMES), "T": np.int64(T)}
        total = 0

        for k in range(n_folds):
            for split in SPLITS:
                idx = wins["fold%d_%s" % (k, split)]
                X = np.empty((idx.shape[0], len(SUMMARY_NAMES)), dtype=np.float32)
                for n, (run_idx, start) in enumerate(idx):
                    o = offsets[run_idx]
                    win = feats[o + start:o + start + T]     # [T, 5, F] raw
                    X[n] = summarize_window(win, feature_names)

                runs = idx[:, 0]
                payload["fold%d_%s_X" % (k, split)] = X
                payload["fold%d_%s_y" % (k, split)] = rule_id[runs]
                payload["fold%d_%s_family" % (k, split)] = family_id[runs]
                payload["fold%d_%s_k" % (k, split)] = neighbor_count[runs]
                payload["fold%d_%s_run" % (k, split)] = runs
                payload["fold%d_%s_exp" % (k, split)] = exp_id[runs]
                total += X.shape[0]

        out = os.path.join(OUT_DIR, "summary_T%d.npz" % T)
        np.savez_compressed(out, **payload)
        print("  T=%3d: %7d windows summarized -> %s" % (T, total, out))

    # ---- how separable is each statistic on its own? -------------------
    # A quick look before any model is fitted. The between-rule spread of a
    # statistic, relative to its within-rule spread, is a crude univariate
    # measure of how much rule information it carries.
    T = 128
    s = np.load(os.path.join(OUT_DIR, "summary_T%d.npz" % T))
    X = np.concatenate([s["fold0_train_X"], s["fold0_val_X"]])
    y = np.concatenate([s["fold0_train_y"], s["fold0_val_y"]])

    print("\nUnivariate separability at T=%d (fold 0 train+val)" % T)
    print("  higher = the statistic alone separates the 11 rules better")
    print("  %-22s %8s" % ("statistic", "F-ratio"))
    scores = []
    for j, name in enumerate(SUMMARY_NAMES):
        col = X[:, j]
        grand = col.mean()
        between = np.mean([(col[y == c].mean() - grand) ** 2 for c in range(11)])
        within = np.mean([col[y == c].var() for c in range(11)])
        scores.append((between / max(within, 1e-12), name))
    for f, name in sorted(scores, reverse=True):
        print("  %-22s %8.3f" % (name, f))


if __name__ == "__main__":
    main()
