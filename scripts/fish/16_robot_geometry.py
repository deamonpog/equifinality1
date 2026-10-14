"""Group geometry of robots vs simulated agents, and the robot-transfer tests.

Produces the numbers of Sect. 6.1 and Table 4 that do not come from a model:

  * mean nearest-neighbor distance per rule (arena radii), simulated agents
    and robots, and the robot / simulated ratio (Table 4, left block; the
    simulated column also gives the k-dependence quoted in Sect. 5.4);
  * the Spearman correlation, across the nine interacting rules, between the
    simulated distance and the robot / simulated ratio;
  * the paired t-tests (over the five folds) of transformer vs GRU robot
    top-1 and top-3 accuracy;
  * the gain in robot top-3 accuracy from expressing time in each domain's
    median kick interval.

The distance is the per-agent, per-event nearest-neighbor distance (the
d_nn1 feature), averaged over the events of a run and then over the runs of a
rule. Simulated agents use the preprocessed runs (burn-in 50 events,
data/fish/runs.npz); robots use the same feature code with the burn-in of the
robot-transfer scripts (10 events).

Inputs:
    data/fish/runs.npz                          (scripts/fish/03)
    FISH_DATA/Robots/*.dat                      (Lei et al. 2020 release)
    results/fish/robot_transfer_comparison.json (scripts/fish/15_gru_robot_transfer.py)

Writes results/fish/robot_geometry.json.

Run from the repository root:
    python scripts/fish/16_robot_geometry.py
"""

import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.getcwd())

from genmod.fish.features import FEATURE_NAMES, compute_features
from genmod.fish.robots import ROBOT_RADIUS, load_robot_runs

RUNS_PATH = os.path.join("data", "fish", "runs.npz")
TRANSFER_PATH = os.path.join("results", "fish", "robot_transfer_comparison.json")
OUT_PATH = os.path.join("results", "fish", "robot_geometry.json")

ROBOT_BURN_IN = 10  # matches scripts/fish/14 and 15
RULE_NAMES = ["no-interaction", "nearest-1", "nearest-2", "nearest-3",
              "random-1", "random-2", "random-3",
              "influential-1", "influential-2", "influential-3", "all-4"]
NN_COL = FEATURE_NAMES.index("d_nn1")


def simulated_distances():
    d = np.load(RUNS_PATH, allow_pickle=True)
    feats, offsets, rule_id = d["feats"], d["offsets"], d["rule_id"]
    per_rule = {}
    for i in range(len(rule_id)):
        x = feats[offsets[i]:offsets[i + 1], :, NN_COL].astype(np.float64)
        per_rule.setdefault(int(rule_id[i]), []).append(x.mean())
    return {r: float(np.mean(v)) for r, v in per_rule.items()}


def robot_distances():
    per_rule = {}
    for run in load_robot_runs():
        f, _ = compute_features(run.time, run.pos, radius=ROBOT_RADIUS,
                                burn_in=ROBOT_BURN_IN)
        per_rule.setdefault(run.rule_id, []).append(
            f[:, :, NN_COL].astype(np.float64).mean())
    return {r: float(np.mean(v)) for r, v in per_rule.items()}


def transfer_tests():
    if not os.path.exists(TRANSFER_PATH):
        print("  %s not found; skipping the paired tests" % TRANSFER_PATH)
        return None
    d = json.load(open(TRANSFER_PATH))
    tr = d["models"]["transformer"]["per_fold"]
    gr = d["models"]["gru"]["per_fold"]
    out = {}
    for key in ("top1", "top3"):
        a = [f["robot_nondim"][key] for f in tr]
        b = [f["robot_nondim"][key] for f in gr]
        t = stats.ttest_rel(a, b)
        out["paired_t_%s" % key] = {
            "transformer_mean": float(np.mean(a)), "gru_mean": float(np.mean(b)),
            "t": float(t.statistic), "p": float(t.pvalue), "n_folds": len(a)}
    for name, folds in (("transformer", tr), ("gru", gr)):
        out["unit_conversion_gain_%s" % name] = {
            key: float(np.mean([f["robot_nondim"][key] - f["robot_seconds"][key]
                                for f in folds]))
            for key in ("top1", "top3")}
    return out


def main():
    print("Nearest-neighbor distance (arena radii)")
    sim = simulated_distances()
    rob = robot_distances()

    rows = []
    print("  %-16s %8s %8s %8s" % ("rule", "sim.", "robots", "ratio"))
    for r in range(len(RULE_NAMES)):
        row = {"rule_id": r, "rule": RULE_NAMES[r], "sim": sim[r],
               "robots": rob.get(r), "ratio": None}
        if r in rob:
            row["ratio"] = rob[r] / sim[r]
            print("  %-16s %8.3f %8.3f %8.2f" % (RULE_NAMES[r], sim[r], rob[r], row["ratio"]))
        else:
            print("  %-16s %8.3f %8s %8s" % (RULE_NAMES[r], sim[r], "-", "-"))
        rows.append(row)

    interacting = [r for r in rob if r != 0]
    rho = stats.spearmanr([sim[r] for r in interacting],
                          [rob[r] / sim[r] for r in interacting])
    print("\nSpearman rho (sim. distance vs ratio, %d interacting rules): %.2f  (p = %.3f)"
          % (len(interacting), rho.statistic, rho.pvalue))

    tests = transfer_tests()
    if tests:
        for key in ("top1", "top3"):
            t = tests["paired_t_%s" % key]
            print("Paired t-test robot %s: transformer %.3f vs GRU %.3f, p = %.3f"
                  % (key, t["transformer_mean"], t["gru_mean"], t["p"]))
        g = tests["unit_conversion_gain_transformer"]
        print("Time-unit conversion gain (transformer): top-1 %+.3f, top-3 %+.3f"
              % (g["top1"], g["top3"]))

    out = {"rows": rows,
           "spearman_rho": float(rho.statistic), "spearman_p": float(rho.pvalue),
           "n_interacting_rules": len(interacting),
           "robot_burn_in": ROBOT_BURN_IN, "transfer_tests": tests}
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nWrote %s" % OUT_PATH)


if __name__ == "__main__":
    main()
