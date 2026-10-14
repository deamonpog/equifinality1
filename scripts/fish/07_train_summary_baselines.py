"""Aggregate-statistic baselines for the fish case study.

Builds the lower rungs of a ladder of representations:

    chance         balanced 11-class floor, 1/11
    rf_naive       random forest on SOLO kinematics only (speed, turning,
                   kick rate, wall distance). No neighbor information at all.
    logreg / rf    the same models on the full group statistics, which are
                   aggregated over agents but ARE relational (nearest-neighbor
                   distance, polarization, cohesion).
    (step 09)      the trajectory Transformer, which keeps each agent as its
                   own token.

The gap between rf_naive and rf measures what the relational features buy.
The gap between rf and the Transformer measures what preserving individual
agent identity buys on top of that (RQ5, ablation 2). Reporting both keeps
the Transformer honest: it has to beat a strong opponent, not a strawman.

Scores are reported as mean +/- std over the five folds. Because every
experiment is tested exactly once across the folds, the fold-to-fold spread
is a genuine estimate of variability across independent simulation seeds, not
a resampling artifact.

Writes results/fish/baselines.json.

Run from the repository root:
    python scripts/fish/07_train_summary_baselines.py
"""

import json
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.getcwd())

from genmod.fish.summary import NAIVE_NAMES, SUMMARY_NAMES

DATA_DIR = os.path.join("data", "fish")
OUT_PATH = os.path.join("results", "fish", "baselines.json")
WINDOW_LENGTHS = [16, 32, 64, 128, 256, 512]
N_FOLDS = 5
N_RULES = 11

# Column positions of the non-relational statistics, for the naive baseline.
NAIVE_COLS = [SUMMARY_NAMES.index(n) for n in NAIVE_NAMES]

# rule_id -> (family_id, k). Mirrors data/fish/rule_lookup.csv.
FAMILY_OF = np.array([0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4])
K_OF = np.array([0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4])


def score(y_true, y_pred):
    """Exact-rule, strategy-family, and neighbor-count metrics.

    The family and k scores are derived from the EXACT prediction rather than
    from separate heads: they ask whether an exact-rule error was a near miss
    (right family, wrong k) or a genuine misunderstanding of the mechanism.
    This is RQ2.
    """
    return {
        "acc": float((y_pred == y_true).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "family_acc": float((FAMILY_OF[y_pred] == FAMILY_OF[y_true]).mean()),
        "k_acc": float((K_OF[y_pred] == K_OF[y_true]).mean()),
        "k_mae": float(np.abs(K_OF[y_pred] - K_OF[y_true]).mean()),
    }


def summarize(per_fold):
    """mean +/- std across folds, for each metric."""
    keys = per_fold[0].keys()
    return {k: (float(np.mean([f[k] for f in per_fold])),
                float(np.std([f[k] for f in per_fold]))) for k in keys}


def main():
    results = {}

    for T in WINDOW_LENGTHS:
        s = np.load(os.path.join(DATA_DIR, "summary_T%d.npz" % T))
        models = {"chance": [], "rf_naive": [], "logreg": [], "rf": []}

        for k in range(N_FOLDS):
            Xtr = s["fold%d_train_X" % k]
            ytr = s["fold%d_train_y" % k]
            Xte = s["fold%d_test_X" % k]
            yte = s["fold%d_test_y" % k]

            # Chance: predict a uniformly random rule. The classes are
            # balanced, so this pins the floor at 1/11 = 0.0909.
            rng = np.random.default_rng(k)
            models["chance"].append(score(yte, rng.integers(0, N_RULES, yte.size)))

            # Naive: solo kinematics only, no neighbor information at all.
            rf_n = RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=k)
            rf_n.fit(Xtr[:, NAIVE_COLS], ytr)
            models["rf_naive"].append(score(yte, rf_n.predict(Xte[:, NAIVE_COLS])))

            # The scaler is fitted on TRAIN ONLY, as everywhere else.
            sc = StandardScaler().fit(Xtr)
            Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

            # Multinomial is the default in current scikit-learn.
            lr = LogisticRegression(max_iter=2000)
            lr.fit(Xtr_s, ytr)
            models["logreg"].append(score(yte, lr.predict(Xte_s)))

            rf = RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=k)
            rf.fit(Xtr, ytr)   # trees do not care about scale
            models["rf"].append(score(yte, rf.predict(Xte)))

        results["T%d" % T] = {m: summarize(v) for m, v in models.items()}

        print("\n=== T = %d kicks (%.0f s of observation)" % (T, T * 0.434))
        print("  %-8s %-14s %-14s %-14s %-14s %-14s"
              % ("model", "exact acc", "macro F1", "family acc", "k acc", "k MAE"))
        for m in ("chance", "rf_naive", "logreg", "rf"):
            r = results["T%d" % T][m]
            print("  %-8s %6.3f +-%.3f %6.3f +-%.3f %6.3f +-%.3f "
                  "%6.3f +-%.3f %6.3f +-%.3f"
                  % (m,
                     r["acc"][0], r["acc"][1],
                     r["macro_f1"][0], r["macro_f1"][1],
                     r["family_acc"][0], r["family_acc"][1],
                     r["k_acc"][0], r["k_acc"][1],
                     r["k_mae"][0], r["k_mae"][1]))

    # ---- the confusion structure of the strongest baseline -------------
    T = 128
    s = np.load(os.path.join(DATA_DIR, "summary_T%d.npz" % T))
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                n_jobs=-1, random_state=0)
    rf.fit(s["fold0_train_X"], s["fold0_train_y"])
    pred = rf.predict(s["fold0_test_X"])
    true = s["fold0_test_y"]

    names = ["none-0", "near-1", "near-2", "near-3", "rand-1", "rand-2",
             "rand-3", "infl-1", "infl-2", "infl-3", "all-4"]
    cm = np.zeros((N_RULES, N_RULES), dtype=int)
    for t, p in zip(true, pred):
        cm[t, p] += 1

    print("\n=== Random forest confusion matrix, T=128, fold 0")
    print("    rows = true rule, columns = predicted")
    print("    %-8s %s" % ("", " ".join("%5s" % n[:5] for n in names)))
    for i, n in enumerate(names):
        row = " ".join("%5d" % c for c in cm[i])
        print("    %-8s %s   (recall %.2f)"
              % (n, row, cm[i, i] / max(cm[i].sum(), 1)))

    # Where do the errors go? Same family, or a different mechanism entirely?
    err = pred != true
    same_family = (FAMILY_OF[pred] == FAMILY_OF[true]) & err
    print("\n    of the %d errors: %d (%.0f%%) stay inside the right family"
          % (err.sum(), same_family.sum(),
             100.0 * same_family.sum() / max(err.sum(), 1)))

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump({"note": "mean, std over 5 grouped folds", "results": results},
                  fh, indent=2)
    print("\nWrote %s" % OUT_PATH)


if __name__ == "__main__":
    main()
