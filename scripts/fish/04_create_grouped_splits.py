"""Assign experiments to grouped cross-validation partitions.

The independent experimental unit is a SIMULATION RUN, and experiment ids are
seed-linked ACROSS rule files (step 01 found 1 experiment with an identical
first event in all 11 files and 13 more with a partially shared one). So the
grouping key is the experiment id, applied simultaneously to all 11 rules:
if experiment 42 is in the test partition, then all eleven runs with
exp_id == 42 are in the test partition.

Without this, a classifier could score well by recognizing the initial
condition rather than the behavioral rule, and nothing in the training curves
would reveal it.

Design: 5-fold cross-validation over the 50 experiments. Each fold holds out
a block of 10 experiments as its test set, so every experiment is tested
exactly once across the five folds. Within a fold:

    test         10 experiments  (this fold's held-out block)
    calibration  10 experiments  (RAPS; must be disjoint from train and test)
    validation    5 experiments  (early stopping, temperature scaling)
    train        25 experiments

Train is only half the experiments, which is the price of an honest test
partition plus a calibration partition large enough for RAPS. It costs less
than it looks: 25 experiments x 11 rules x ~1550 events still yields tens of
thousands of training windows.

Writes data/fish/splits.json.

Run from the repository root:
    python scripts/fish/04_create_grouped_splits.py
"""

import json
import os

import numpy as np

OUT_PATH = os.path.join("data", "fish", "splits.json")
RUNS_PATH = os.path.join("data", "fish", "runs.npz")

N_FOLDS = 5
SEED = 20260713

# Experiments per partition, within each fold. N_TEST is fixed by the fold
# structure: the held-out block IS the test set, which is what makes every
# experiment tested exactly once. The rest are drawn from the other folds.
N_TEST = 10   # = n_exp / N_FOLDS
N_CALIB = 10
N_VAL = 5


def main():
    if not os.path.exists(RUNS_PATH):
        raise SystemExit("Missing %s. Run 03 first." % RUNS_PATH)

    data = np.load(RUNS_PATH, allow_pickle=True)
    exp_ids = np.unique(data["exp_id"])
    rule_ids = data["rule_id"]
    n_exp = exp_ids.size
    print("Experiments: %d   runs: %d   rules: %d"
          % (n_exp, data["exp_id"].size, np.unique(rule_ids).size))

    if N_TEST + N_CALIB + N_VAL >= n_exp:
        raise SystemExit("partition sizes exceed the number of experiments")

    # A single fixed permutation of the experiments defines the folds. Each
    # fold takes a contiguous block of the permutation as its held-out set,
    # so every experiment lands in exactly one fold's test/calibration block.
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(exp_ids)
    blocks = np.array_split(perm, N_FOLDS)  # 5 blocks of 10 experiments

    folds = []
    for k in range(N_FOLDS):
        # The held-out block is the test set in full. This is the property
        # that makes every experiment tested exactly once over the 5 folds.
        test = blocks[k]
        if test.size != N_TEST:
            raise SystemExit("fold %d has %d test experiments, expected %d"
                             % (k, test.size, N_TEST))

        # The other 40 experiments supply calibration, validation, and train.
        # Calibration must be sizeable: with n_cal points the RAPS quantile
        # sits at ceil((n+1)(1-alpha))/n, which is only attainable when
        # n >= 1/alpha - 1. At alpha = 0.05 that is a hard floor of 19
        # CALIBRATION WINDOWS. We will have far more, but a larger calibration
        # partition also means a less noisy quantile.
        rest = np.array([e for e in perm if e not in set(test.tolist())])
        rest = rng.permutation(rest)

        calib = rest[:N_CALIB]
        val = rest[N_CALIB:N_CALIB + N_VAL]
        train = rest[N_CALIB + N_VAL:]

        folds.append({
            "fold": k,
            "train": sorted(int(e) for e in train),
            "val": sorted(int(e) for e in val),
            "calib": sorted(int(e) for e in calib),
            "test": sorted(int(e) for e in test),
        })

    # ---- the assertions that protect the whole study -------------------
    print("\n[1] Leakage assertions")
    all_exp = set(int(e) for e in exp_ids)
    for f in folds:
        parts = {p: set(f[p]) for p in ("train", "val", "calib", "test")}
        names = list(parts)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                overlap = parts[names[i]] & parts[names[j]]
                assert not overlap, (
                    "fold %d: experiments %s appear in both %s and %s"
                    % (f["fold"], sorted(overlap), names[i], names[j]))
        union = set().union(*parts.values())
        assert union == all_exp, (
            "fold %d: partitions do not cover all experiments" % f["fold"])
    print("    no experiment appears in two partitions of a fold ... True")
    print("    every experiment is assigned in every fold .......... True")

    # Every experiment is tested exactly once across the five folds.
    test_counts = {int(e): 0 for e in exp_ids}
    for f in folds:
        for e in f["test"]:
            test_counts[e] += 1
    once = all(c == 1 for c in test_counts.values())
    print("    every experiment is tested exactly once ............. %s" % once)
    assert once

    # ---- class balance within each partition ---------------------------
    # Because the grouping is by experiment and every experiment exists under
    # all 11 rules, each partition is automatically balanced: n_experiments
    # runs of every rule. Verify rather than assume.
    print("\n[2] Runs per partition (should be n_experiments x 11 rules)")
    print("    %-5s %-24s %-24s %-24s %-24s"
          % ("fold", "train", "val", "calib", "test"))
    for f in folds:
        cells = []
        for p in ("train", "val", "calib", "test"):
            sel = np.isin(data["exp_id"], f[p])
            per_rule = np.bincount(rule_ids[sel], minlength=11)
            balanced = len(set(per_rule.tolist())) == 1
            cells.append("%3d runs (%2d/rule) %s"
                         % (sel.sum(), per_rule[0], "ok" if balanced else "SKEW"))
        print("    %-5d %-24s %-24s %-24s %-24s" % (f["fold"], *cells))

    with open(OUT_PATH, "w") as fh:
        json.dump({
            "n_folds": N_FOLDS,
            "seed": SEED,
            "grouping_key": "exp_id",
            "note": ("Experiment ids are seed-linked across rule files, so an "
                     "experiment is assigned to the same partition in all 11 "
                     "rules. Windows inherit the partition of their run."),
            "folds": folds,
        }, fh, indent=2)

    print("\n[3] Fold 0 assignment (experiments)")
    f = folds[0]
    for p in ("train", "val", "calib", "test"):
        print("    %-6s (%2d) %s" % (p, len(f[p]), f[p]))

    print("\nWrote %s" % OUT_PATH)


if __name__ == "__main__":
    main()
