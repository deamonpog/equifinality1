"""Full evaluation of a trained fish rule classifier.

Consumes the probability files written by step 08 or 09 and reports every
metric the case study needs. Nothing here retrains anything, so the analysis
can be iterated freely.

    exact rule        top-1, top-3, macro-F1, balanced accuracy, confusion
    strategy family   accuracy and macro-F1 over {none, nearest, random,
                      mostinfluential, all}
    neighbor count    accuracy and mean absolute error over k
    probabilistic     NLL, multiclass Brier score, expected calibration error
    error structure   do errors preserve the family, or the neighbor count?

All confidence intervals resample RUNS, never windows: windows drawn from the
same simulation run are strongly dependent, and treating ~15 of them as
independent observations would shrink the intervals by roughly a factor of 4.

Usage:
    python scripts/fish/11_evaluate_fish_rules.py --tag transformer_T64
    python scripts/fish/11_evaluate_fish_rules.py --tag transformer_T64 --compare gru_T64
"""

import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

sys.path.insert(0, os.getcwd())

from genmod.evaluation.calibration import brier_score, expected_calibration_error

RULE_NAMES = ["none-0", "near-1", "near-2", "near-3", "rand-1", "rand-2",
              "rand-3", "infl-1", "infl-2", "infl-3", "all-4"]
FAMILY_NAMES = ["none", "nearest", "random", "mostinfluential", "all"]
FAMILY_OF = np.array([0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4])
K_OF = np.array([0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4])
N_BOOT = 2000


def load_tag(tag):
    """Pool the test predictions of every fold. Run ids are offset per fold so
    a bootstrap can never merge two different folds' runs."""
    d = os.path.join("results", "fish", tag)
    folds = sorted(int(f[len("probs_fold"):-len(".npz")])
                   for f in os.listdir(d) if f.startswith("probs_fold"))
    P, Y, R = [], [], []
    for k in folds:
        z = np.load(os.path.join(d, "probs_fold%d.npz" % k))
        P.append(z["test_probs"])
        Y.append(z["test_labels"])
        R.append(z["test_runs"] + 10000 * k)
    return np.concatenate(P), np.concatenate(Y), np.concatenate(R), folds


def boot_ci(fn, probs, labels, runs, n_boot=N_BOOT, seed=0):
    """95% CI of a metric, resampling whole RUNS with replacement."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(runs)
    idx_by_run = {r: np.where(runs == r)[0] for r in uniq}
    vals = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.choice(uniq, size=uniq.size, replace=True)
        sel = np.concatenate([idx_by_run[r] for r in picked])
        vals[b] = fn(probs[sel], labels[sel])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def top1(p, y):
    return float((p.argmax(1) == y).mean())


def top3(p, y):
    top = np.argsort(-p, axis=1)[:, :3]
    return float(np.mean([y[i] in top[i] for i in range(len(y))]))


def macro_f1(p, y):
    return float(f1_score(y, p.argmax(1), average="macro"))


def family_acc(p, y):
    return float((FAMILY_OF[p.argmax(1)] == FAMILY_OF[y]).mean())


def k_acc(p, y):
    return float((K_OF[p.argmax(1)] == K_OF[y]).mean())


def k_mae(p, y):
    return float(np.abs(K_OF[p.argmax(1)] - K_OF[y]).mean())


def nll(p, y):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())


def report(tag):
    probs, labels, runs, folds = load_tag(tag)
    print("\n" + "=" * 74)
    print("%s   %d test windows from %d runs, folds %s"
          % (tag, len(labels), len(np.unique(runs)), folds))
    print("=" * 74)

    metrics = {}
    print("\n[1] Exact-rule identification (the primary task, chance = 0.0909)")
    for name, fn in [("top-1 accuracy", top1), ("top-3 accuracy", top3),
                     ("macro F1", macro_f1)]:
        v = fn(probs, labels)
        lo, hi = boot_ci(fn, probs, labels, runs)
        metrics[name] = [v, lo, hi]
        print("    %-18s %.4f   95%% CI [%.4f, %.4f]" % (name, v, lo, hi))
    bal = float(balanced_accuracy_score(labels, probs.argmax(1)))
    metrics["balanced accuracy"] = [bal, None, None]
    print("    %-18s %.4f" % ("balanced accuracy", bal))

    print("\n[2] Hierarchical identification (RQ2)")
    for name, fn in [("family accuracy", family_acc), ("k accuracy", k_acc),
                     ("k MAE", k_mae)]:
        v = fn(probs, labels)
        lo, hi = boot_ci(fn, probs, labels, runs)
        metrics[name] = [v, lo, hi]
        print("    %-18s %.4f   95%% CI [%.4f, %.4f]" % (name, v, lo, hi))

    print("\n[3] Probabilistic quality")
    for name, fn in [("NLL", nll),
                     ("Brier", lambda p, y: float(brier_score(p, y))),
                     # expected_calibration_error returns
                     # (ece, bin_accs, bin_confs, bin_counts); we want the ece.
                     ("ECE", lambda p, y: float(expected_calibration_error(p, y)[0]))]:
        v = fn(probs, labels)
        metrics[name] = [v, None, None]
        print("    %-18s %.4f" % (name, v))

    print("\n[4] Confusion matrix (rows = true, columns = predicted)")
    pred = probs.argmax(1)
    cm = np.zeros((11, 11), dtype=int)
    for t, p in zip(labels, pred):
        cm[t, p] += 1
    print("    %-9s%s" % ("", " ".join("%5s" % n for n in RULE_NAMES)))
    for i, n in enumerate(RULE_NAMES):
        print("    %-9s%s   recall %.2f"
              % (n, " ".join("%5d" % c for c in cm[i]),
                 cm[i, i] / max(cm[i].sum(), 1)))

    print("\n[5] Where do the errors go? (this is the RQ2 story)")
    err = pred != labels
    n_err = int(err.sum())
    same_fam = int(((FAMILY_OF[pred] == FAMILY_OF[labels]) & err).sum())
    same_k = int(((K_OF[pred] == K_OF[labels]) & err).sum())
    neither = int((err & (FAMILY_OF[pred] != FAMILY_OF[labels])
                   & (K_OF[pred] != K_OF[labels])).sum())
    print("    total errors ....................... %d (%.1f%% of windows)"
          % (n_err, 100.0 * n_err / len(labels)))
    if n_err:
        print("    right family, wrong k .............. %d (%.0f%% of errors)"
              % (same_fam, 100.0 * same_fam / n_err))
        print("    right k, wrong family .............. %d (%.0f%% of errors)"
              % (same_k, 100.0 * same_k / n_err))
        print("    both wrong ......................... %d (%.0f%% of errors)"
              % (neither, 100.0 * neither / n_err))
        if same_k > same_fam:
            print("    => the NUMBER of influencing neighbors is recovered more")
            print("       reliably than WHICH neighbors are selected.")
        else:
            print("    => the strategy family is recovered more reliably than k.")

    print("\n[6] Most confused rule pairs")
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = [(off[i, j], i, j) for i in range(11) for j in range(11) if off[i, j]]
    for c, i, j in sorted(pairs, reverse=True)[:8]:
        tag_rel = ("same family" if FAMILY_OF[i] == FAMILY_OF[j]
                   else ("same k" if K_OF[i] == K_OF[j] else "unrelated"))
        print("    %-8s -> %-8s  %4d  (%s)"
              % (RULE_NAMES[i], RULE_NAMES[j], c, tag_rel))

    return metrics, probs, labels, runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="transformer_T64")
    ap.add_argument("--compare", nargs="*", default=[],
                    help="other tags to compare against, e.g. gru_T64")
    args = ap.parse_args()

    metrics, probs, labels, runs = report(args.tag)

    for other in args.compare:
        m2, p2, y2, r2 = report(other)

        # A paired comparison on the SAME windows, bootstrapped over runs, is
        # the honest way to ask whether the difference is real. Comparing two
        # independent confidence intervals would be far more conservative and
        # would ignore that both models saw identical data.
        if len(y2) == len(labels) and np.array_equal(y2, labels):
            rng = np.random.default_rng(0)
            uniq = np.unique(runs)
            idx_by_run = {r: np.where(runs == r)[0] for r in uniq}
            diffs = np.empty(N_BOOT)
            for b in range(N_BOOT):
                picked = rng.choice(uniq, size=uniq.size, replace=True)
                sel = np.concatenate([idx_by_run[r] for r in picked])
                diffs[b] = top1(probs[sel], labels[sel]) - top1(p2[sel], labels[sel])
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            print("\n" + "=" * 74)
            print("Paired comparison: %s minus %s" % (args.tag, other))
            print("  delta top-1 accuracy  %+.4f   95%% CI [%+.4f, %+.4f]"
                  % (diffs.mean(), lo, hi))
            print("  the difference is %s at the 5%% level"
                  % ("REAL" if lo > 0 or hi < 0 else "not distinguishable"))

    out = os.path.join("results", "fish", args.tag, "evaluation.json")
    with open(out, "w") as fh:
        json.dump({k: v for k, v in metrics.items()}, fh, indent=2)
    print("\nWrote %s" % out)


if __name__ == "__main__":
    main()
