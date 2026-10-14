"""RAPS conformal prediction for the fish case study.

Consumes the probability files written by step 09 and produces prediction
sets with a distribution-free marginal coverage guarantee:

    P(g_true in C_alpha(W))  >=  1 - alpha

The calibration split has been touched by nothing else: the model was trained
on train, early-stopped and temperature-scaled on validation, and the
calibration experiments were never seen. Its only job is to set the conformal
quantile.

Two statistical points specific to this data.

Exchangeability. Windows drawn from the same run are strongly dependent, so
calibration and test windows are NOT i.i.d. draws. Two things keep the
guarantee meaningful: calibration and test windows come from DISJOINT
experiments (so no run contributes to both), and the eval windows are
NON-OVERLAPPING within a run (step 05). Coverage is therefore marginal over
the window distribution induced by exchangeable RUNS.

Confidence intervals. Bootstrapping over windows would treat ~15 windows from
one run as 15 independent observations and produce intervals several times too
narrow. All intervals here resample RUNS with replacement.

Usage:
    python scripts/fish/10_calibrate_fish_raps.py --tag transformer_T64
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

from genmod.evaluation.conformal import (compute_nonconformity_scores,
                                         conformal_calibrate,
                                         conformal_predict)

RULE_NAMES = ["none-0", "near-1", "near-2", "near-3", "rand-1", "rand-2",
              "rand-3", "infl-1", "infl-2", "infl-3", "all-4"]
FAMILY_OF = np.array([0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4])
K_OF = np.array([0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4])
N_BOOT = 2000


def bootstrap_ci(values, runs, n_boot=N_BOOT, seed=0):
    """95% CI for the mean of a per-window quantity, resampling RUNS.

    The window is not the experimental unit. Resampling windows would ignore
    the within-run dependence and shrink the interval by roughly sqrt(windows
    per run).
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(runs)
    by_run = {r: values[runs == r] for r in uniq}
    means = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.choice(uniq, size=uniq.size, replace=True)
        means[b] = np.concatenate([by_run[r] for r in picked]).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="transformer_T64")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    ap.add_argument("--k-reg", type=int, default=2)
    ap.add_argument("--lambda-reg", type=float, default=0.1)
    args = ap.parse_args()

    res_dir = os.path.join("results", "fish", args.tag)
    folds = sorted(int(f[len("probs_fold"):-len(".npz")])
                   for f in os.listdir(res_dir) if f.startswith("probs_fold"))
    print("Tag %s, folds %s" % (args.tag, folds))

    # Seed the RAPS tie-breaking randomization so the sets are reproducible.
    np.random.seed(20260713)

    out = {"tag": args.tag, "k_reg": args.k_reg, "lambda_reg": args.lambda_reg,
           "alphas": {}}

    for alpha in args.alphas:
        cover_all, size_all, runs_all, labels_all = [], [], [], []
        qhats = []

        for k in folds:
            d = np.load(os.path.join(res_dir, "probs_fold%d.npz" % k))
            cal_p, cal_y = d["calib_probs"], d["calib_labels"]
            te_p, te_y = d["test_probs"], d["test_labels"]
            te_runs = d["test_runs"]

            scores = compute_nonconformity_scores(
                cal_p, cal_y, method="raps",
                k_reg=args.k_reg, lambda_reg=args.lambda_reg)
            qhat = conformal_calibrate(scores, alpha=alpha)
            qhats.append(qhat)

            sets = conformal_predict(
                te_p, qhat, method="raps",
                k_reg=args.k_reg, lambda_reg=args.lambda_reg)

            cover_all.append(np.array([y in s for s, y in zip(sets, te_y)],
                                      dtype=float))
            size_all.append(np.array([len(s) for s in sets], dtype=float))
            # Offset run ids per fold so bootstrap never merges runs across folds.
            runs_all.append(te_runs + 10000 * k)
            labels_all.append(te_y)

        cover = np.concatenate(cover_all)
        size = np.concatenate(size_all)
        runs = np.concatenate(runs_all)
        labels = np.concatenate(labels_all)

        cov_lo, cov_hi = bootstrap_ci(cover, runs)
        sz_lo, sz_hi = bootstrap_ci(size, runs)

        per_class = {}
        for c in range(11):
            m = labels == c
            per_class[RULE_NAMES[c]] = {
                "coverage": float(cover[m].mean()),
                "mean_set_size": float(size[m].mean()),
                "n": int(m.sum()),
            }

        sizes_hist = {int(s): int((size == s).sum()) for s in np.unique(size)}

        out["alphas"]["%.2f" % alpha] = {
            "target_coverage": 1 - alpha,
            "empirical_coverage": float(cover.mean()),
            "coverage_ci95": [cov_lo, cov_hi],
            "mean_set_size": float(size.mean()),
            "set_size_ci95": [sz_lo, sz_hi],
            "median_set_size": float(np.median(size)),
            "singleton_rate": float((size == 1).mean()),
            "mean_qhat": float(np.mean(qhats)),
            "set_size_hist": sizes_hist,
            "per_class": per_class,
        }

        print("\n=== alpha = %.2f   (target coverage %.2f)" % (alpha, 1 - alpha))
        print("    empirical coverage   %.4f   95%% CI [%.4f, %.4f]"
              % (cover.mean(), cov_lo, cov_hi))
        print("    mean set size        %.3f   95%% CI [%.3f, %.3f]"
              % (size.mean(), sz_lo, sz_hi))
        print("    median set size      %.1f" % np.median(size))
        print("    singleton rate       %.3f" % (size == 1).mean())
        covered = cov_lo <= (1 - alpha) <= cov_hi or cover.mean() >= 1 - alpha
        print("    achieves target      %s" % covered)

        print("    set-size distribution:")
        for s in sorted(sizes_hist):
            n = sizes_hist[s]
            print("      |C| = %2d  %5d windows (%5.1f%%)  %s"
                  % (s, n, 100.0 * n / size.size, "#" * int(60.0 * n / size.size)))

        print("    per-rule coverage:")
        for c in range(11):
            pc = per_class[RULE_NAMES[c]]
            flag = "" if pc["coverage"] >= 1 - alpha - 0.05 else "  <- under"
            print("      %-8s cov %.3f   mean |C| %.2f%s"
                  % (RULE_NAMES[c], pc["coverage"], pc["mean_set_size"], flag))

    path = os.path.join(res_dir, "conformal.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nWrote %s" % path)


if __name__ == "__main__":
    main()
