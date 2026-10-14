"""Capacity / depth statistics for the fish rule-identification task (Table 2).

Recomputes, from the VERIFIED probability files, everything the manuscript will
say about model capacity. Nothing about capacity should be quoted from a commit
message or a training-log tail; it comes from this script's committed JSON.

Four models enter the analysis (roles, with the shorthand used below):

    small   ~148k transformer, 2 layers, per-agent invariant tokens      A_s
    depth   ~183k transformer, 4 layers, per-agent invariant tokens      A_d
    large    1.8M transformer, 4 layers, per-agent invariant tokens      A_L
    gru     ~156k GRU over 13 per-event group statistics                 A_g

Two labelled within-family comparisons:

    A_d - A_s   comparable-budget DEPTH/ARCHITECTURE comparison
                (budget ~fixed; depth and the width needed to hit it both move)
    A_d - A_L   WIDTH / PARAMETER-BUDGET comparison at FIXED depth

and the normalized gap-recovery statistic

    g = (A_d - A_s) / (A_L - A_s)

The GRU comparisons (A_g - A_L, A_g - A_s) are a REPRESENTATIONAL-BURDEN
contrast, confounded by input representation; they are reported, not used to
claim architecture superiority.

Bootstrap clustering
--------------------
The cross-validation folds are grouped by exp_id (50 seed-linked initial
conditions, each spanning all 11 rules), so same-experiment/different-rule
trajectories are dependent. Therefore:

    PRIMARY      resample whole EXPERIMENTS  (exp_id, ~50 clusters)
    SENSITIVITY  resample whole RUNS         (run_index, ~550 clusters)

The run-level interval is reported alongside the primary one, never as the
headline.

Point estimates are full-data (not bootstrap means). Confidence intervals are
percentile intervals of the paired bootstrap; the SAME resample draw indexes
every model each iteration, so the pairing is exact.

Practical-equivalence branch verdict
-------------------------------------
A result is "near" an endpoint only when the FULL paired-difference CI lies
inside [-delta, +delta] (TOST-style), never merely because the CI includes
zero. Reported over delta in {0.01, 0.02, 0.03} so the conclusion's dependence
on the margin is explicit.

Usage:
    python scripts/fish/12_capacity_stats.py \
        --small transformer_inv_T64_small_verify \
        --depth transformer_inv_T64_depth \
        --large transformer_inv_T64 \
        --gru   gru_T64
Any role whose results directory is absent is skipped, and the statistics that
need it are omitted (so the script can be smoke-tested against existing tags).
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

N_BOOT = 2000
SEED = 0
DELTAS = [0.01, 0.02, 0.03]
DATA_DIR = os.path.join("data", "fish")


def top1(probs, labels, sel):
    return float((probs[sel].argmax(1) == labels[sel]).mean())


def load_tag(tag, exp_id_arr):
    """Pool every fold's test predictions for one tag.

    Returns probs [N,C], labels [N], exp_ids [N] (experiment of each window),
    pooled_run [N] (run_index offset per fold so folds never merge), and the
    fold list. exp_ids come from runs.npz indexed by the raw run_index that the
    training script stored as test_runs.
    """
    d = os.path.join("results", "fish", tag)
    if not os.path.isdir(d):
        return None
    fold_files = [f for f in os.listdir(d)
                  if f.startswith("probs_fold") and f.endswith(".npz")]
    if not fold_files:
        return None
    folds = sorted(int(f[len("probs_fold"):-len(".npz")]) for f in fold_files)
    P, Y, E, R = [], [], [], []
    for k in folds:
        z = np.load(os.path.join(d, "probs_fold%d.npz" % k))
        raw_run = z["test_runs"]
        P.append(z["test_probs"])
        Y.append(z["test_labels"])
        E.append(exp_id_arr[raw_run])
        R.append(raw_run + 10000 * k)
    return {
        "tag": tag,
        "probs": np.concatenate(P),
        "labels": np.concatenate(Y),
        "exp": np.concatenate(E),
        "run": np.concatenate(R),
        "folds": folds,
        "n_params": read_n_params(d),
    }


def read_n_params(run_dir):
    """Exact parameter count, from run_summary.json if present, else the npz."""
    summ = os.path.join(run_dir, "run_summary.json")
    if os.path.exists(summ):
        with open(summ) as fh:
            v = json.load(fh).get("n_params")
            if v is not None:
                return int(v)
    for f in sorted(os.listdir(run_dir)):
        if f.startswith("probs_fold") and f.endswith(".npz"):
            z = np.load(os.path.join(run_dir, f))
            if "n_params" in z:
                return int(z["n_params"])
            break
    return None


def stat_vector(models, labels, sel):
    """All named quantities on one index selection, as an ordered dict.

    `models` maps role -> probs array. Only quantities whose inputs are present
    are included, so the same code serves the smoke test and the full run.
    """
    acc = {role: top1(m, labels, sel) for role, m in models.items()}
    out = {}
    for role in ("small", "depth", "large", "gru"):
        if role in acc:
            out["acc_%s" % role] = acc[role]
    if "depth" in acc and "small" in acc:
        out["d_depth_minus_small"] = acc["depth"] - acc["small"]
    if "depth" in acc and "large" in acc:
        out["d_depth_minus_large"] = acc["depth"] - acc["large"]
    if "large" in acc and "small" in acc:
        out["d_large_minus_small"] = acc["large"] - acc["small"]
    if "gru" in acc and "large" in acc:
        out["d_gru_minus_large"] = acc["gru"] - acc["large"]
    if "gru" in acc and "small" in acc:
        out["d_gru_minus_small"] = acc["gru"] - acc["small"]
    if {"small", "depth", "large"} <= set(acc):
        denom = acc["large"] - acc["small"]
        out["g_gap_recovery"] = ((acc["depth"] - acc["small"]) / denom
                                 if abs(denom) > 1e-6 else float("nan"))
    return out


def cluster_bootstrap(cluster_ids, names, models, labels, n_boot=N_BOOT, seed=SEED):
    """Percentile CIs for every named quantity, resampling whole clusters.

    One resample draw per iteration indexes every model (exact pairing).
    Returns {name: (lo, hi)}.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(cluster_ids)
    idx_by = {c: np.where(cluster_ids == c)[0] for c in uniq}
    samples = {n: np.empty(n_boot) for n in names}
    for b in range(n_boot):
        picked = rng.choice(uniq, size=uniq.size, replace=True)
        sel = np.concatenate([idx_by[c] for c in picked])
        v = stat_vector(models, labels, sel)
        for n in names:
            samples[n][b] = v.get(n, float("nan"))
    ci = {}
    for n in names:
        s = samples[n][~np.isnan(samples[n])]
        ci[n] = ((float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5)))
                 if s.size else (float("nan"), float("nan")))
    return ci


def within(ci, delta):
    lo, hi = ci
    return (not np.isnan(lo)) and lo >= -delta and hi <= delta


def outside(ci, delta):
    lo, hi = ci
    return (not np.isnan(lo)) and (lo > delta or hi < -delta)


def branch_verdict(ci_ds, ci_dL, delta):
    """Pre-registered practical-equivalence verdict for the ~183k/4L result.

    ci_ds = CI of (A_d - A_s); ci_dL = CI of (A_d - A_L).
    """
    near_small = within(ci_ds, delta)
    near_large = within(ci_dL, delta)
    if near_small and near_large:
        verdict = "inconclusive_endpoints_within_margin"
    elif near_small:
        verdict = "near_small_baseline"
    elif near_large:
        verdict = "near_large_model"
    elif outside(ci_ds, delta) and outside(ci_dL, delta):
        verdict = "intermediate"
    else:
        verdict = "inconclusive"
    interp = {
        "near_small_baseline":
            ("comparable-budget depth/architecture change yields no practically "
             "meaningful improvement in extraction performance within the tested "
             "configurations"),
        "near_large_model":
            ("~1/10 the parameter budget at full depth practically recovers the "
             "large model's extraction"),
        "intermediate":
            ("A_d differs meaningfully from both endpoints; both depth/architecture "
             "and width/budget contribute -- report g with CI"),
        "inconclusive":
            ("a paired-difference CI straddles a delta boundary; underpowered at "
             "this margin"),
        "inconclusive_endpoints_within_margin":
            ("both endpoints lie within delta of A_d; the endpoints are themselves "
             "practically equivalent at this margin"),
    }[verdict]
    return {"verdict": verdict, "near_small": near_small,
            "near_large": near_large, "interpretation": interp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", default="transformer_inv_T64_small_verify")
    ap.add_argument("--depth", default="transformer_inv_T64_depth")
    ap.add_argument("--large", default="transformer_inv_T64")
    ap.add_argument("--gru", default="gru_T64")
    ap.add_argument("--out", default=os.path.join("results", "fish",
                                                  "capacity_stats.json"))
    ap.add_argument("--n_boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    exp_id_arr = np.load(os.path.join(DATA_DIR, "runs.npz"),
                         allow_pickle=True)["exp_id"]

    loaded = {}
    for role, tag in [("small", args.small), ("depth", args.depth),
                      ("large", args.large), ("gru", args.gru)]:
        d = load_tag(tag, exp_id_arr)
        if d is None:
            print("  skip %-6s (%s): results not found" % (role, tag))
        else:
            loaded[role] = d
            print("  load %-6s (%s): %d windows, %d exp, %d run, params=%s"
                  % (role, tag, len(d["labels"]), len(np.unique(d["exp"])),
                     len(np.unique(d["run"])), d["n_params"]))

    if len(loaded) < 2:
        sys.exit("Need at least two model roles present to compare; found %d."
                 % len(loaded))

    # Alignment: every model must classify the SAME windows in the SAME order,
    # or the paired bootstrap is meaningless. Assert on both labels and the
    # (fold-offset) run ids -- the existing evaluate script only checked labels.
    ref = next(iter(loaded.values()))
    labels = ref["labels"]
    for role, d in loaded.items():
        if not (len(d["labels"]) == len(labels)
                and np.array_equal(d["labels"], labels)):
            sys.exit("Label vectors differ (%s vs %s): windows are not aligned; "
                     "paired comparison invalid." % (role, ref["tag"]))
        if not np.array_equal(d["run"], ref["run"]):
            sys.exit("Run ids differ (%s vs %s): window ordering is not aligned; "
                     "paired comparison invalid." % (role, ref["tag"]))
    exp_ids = ref["exp"]
    run_ids = ref["run"]
    models = {role: d["probs"] for role, d in loaded.items()}

    all_idx = np.arange(len(labels))
    point = stat_vector(models, labels, all_idx)
    names = list(point.keys())

    boot = {
        "exp_id": cluster_bootstrap(exp_ids, names, models, labels, args.n_boot),
        "run": cluster_bootstrap(run_ids, names, models, labels, args.n_boot),
    }

    # Branch verdict uses the PRIMARY (exp_id) clustering.
    delta_sens = {}
    if "d_depth_minus_small" in names and "d_depth_minus_large" in names:
        for delta in DELTAS:
            delta_sens["%.2f" % delta] = branch_verdict(
                boot["exp_id"]["d_depth_minus_small"],
                boot["exp_id"]["d_depth_minus_large"], delta)

    result = {
        "meta": {
            "n_boot": args.n_boot, "seed": SEED,
            "clustering_primary": "exp_id", "clustering_sensitivity": "run",
            "n_experiments": int(len(np.unique(exp_ids))),
            "n_runs": int(len(np.unique(run_ids))),
            "n_windows": int(len(labels)),
            "deltas": DELTAS,
            "point_estimate": "full-data (not bootstrap mean)",
        },
        "tags": {role: {"tag": d["tag"], "n_params": d["n_params"],
                        "folds": d["folds"]}
                 for role, d in loaded.items()},
        "point": point,
        "bootstrap": {clust: {n: {"point": point[n], "ci_lo": ci[n][0],
                                  "ci_hi": ci[n][1]}
                              for n in names}
                      for clust, ci in boot.items()},
        "delta_sensitivity_primary": delta_sens,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)

    # Human-readable summary.
    print("\n" + "=" * 74)
    print("Capacity statistics  (primary = exp_id-clustered; run = sensitivity)")
    print("=" * 74)
    for role in ("small", "depth", "large", "gru"):
        if role in loaded:
            print("  A_%-5s %-34s acc %.4f  (%s params)"
                  % ({"small": "s", "depth": "d", "large": "L", "gru": "g"}[role],
                     loaded[role]["tag"], point["acc_%s" % role],
                     loaded[role]["n_params"]))
    print("\n  paired differences (full-data point; 95%% CI exp_id | run):")
    for n in names:
        if n.startswith("d_") or n == "g_gap_recovery":
            e, r = boot["exp_id"][n], boot["run"][n]
            print("    %-22s %+.4f   exp[%+.4f,%+.4f]  run[%+.4f,%+.4f]"
                  % (n, point[n], e[0], e[1], r[0], r[1]))
    if delta_sens:
        print("\n  practical-equivalence verdict for A_d (~183k/4L), by margin:")
        for k in sorted(delta_sens):
            v = delta_sens[k]
            print("    delta=%s  -> %-38s (near_small=%s near_large=%s)"
                  % (k, v["verdict"], v["near_small"], v["near_large"]))
    print("\nWrote %s" % args.out)


if __name__ == "__main__":
    main()
