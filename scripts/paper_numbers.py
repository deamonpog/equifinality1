"""Check every reported number of the paper against the result files.

Reads only files under results/ (no data, no GPU, no retraining) and prints
each number the paper reports next to the value recomputed from those files.

    match     the recomputed value rounds to the reported value
    ~         off by one unit in the last reported digit (a rounding boundary)
    MISMATCH  anything else

Exit status is 1 if any number is a MISMATCH.

Run from the repository root:
    python scripts/paper_numbers.py
"""

import json
import os
import sys

import numpy as np

RES = "results"
FISH = os.path.join(RES, "fish")
ROWS = []

FAMILY_OF = np.array([0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4])
K_OF = np.array([0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4])
RULES = ["none-0", "near-1", "near-2", "near-3", "rand-1", "rand-2",
         "rand-3", "infl-1", "infl-2", "infl-3", "all-4"]


def load(*parts):
    with open(os.path.join(*parts)) as fh:
        return json.load(fh)


def check(where, what, reported, value, digits):
    """Compare a recomputed value with the number printed in the paper."""
    unit = 10.0 ** (-digits)
    diff = abs(round(value, digits) - reported)
    if diff < unit / 2:
        status = "match"
    elif diff < 1.5 * unit:
        status = "~"
    else:
        status = "MISMATCH"
    ROWS.append((where, what, reported, value, digits, status))


def check_below(where, what, bound, value, digits):
    """The paper states an upper bound rather than a value."""
    ROWS.append((where, what + " (< %g)" % bound, bound, value, digits,
                 "match" if value < bound else "MISMATCH"))


def pooled(tag):
    d = os.path.join(FISH, tag)
    folds = sorted(int(f[10:-4]) for f in os.listdir(d) if f.startswith("probs_fold"))
    P, Y = [], []
    for k in folds:
        z = np.load(os.path.join(d, "probs_fold%d.npz" % k))
        P.append(z["test_probs"])
        Y.append(z["test_labels"])
    return np.concatenate(P), np.concatenate(Y)


def schelling():
    base = load(RES, "logs", "ruletree_base", "results.json")
    rowc = load(RES, "logs", "ruletree_rowcol", "results.json")
    cb = load(RES, "logs", "ruletree_base", "conformal_recalibrated.json")["results"]
    cr = load(RES, "logs", "ruletree_rowcol", "conformal_recalibrated.json")["results"]
    w = "Sect. 4"
    check(w, "library size", 2089, base["num_classes"], 0)
    check(w, "chance level (%)", 0.048, 100.0 / base["num_classes"], 3)
    check(w, "top-1 accuracy", 0.319, base["test_top1"], 3)
    check(w, "top-3 accuracy", 0.485, base["test_top3"], 3)
    check(w, "top-5 accuracy", 0.562, base["test_top5"], 3)
    check(w, "ECE before temperature scaling", 0.017, base["ece_pre_temp"], 3)
    w = "Table 1"
    for a, cov, size in (("0.01", 0.989, 234.3), ("0.05", 0.948, 104.2),
                         ("0.1", 0.897, 52.9), ("0.2", 0.801, 20.5)):
        check(w, "coverage, alpha=%s" % a, cov, cb[a]["coverage"], 3)
        check(w, "mean |C|, alpha=%s" % a, size, cb[a]["avg_set_size"], 1)
        check(w, "%% of library, alpha=%s" % a, round(100 * size / 2089, 1),
              100 * cb[a]["avg_set_size"] / 2089, 1)
    check(w, "row+col mean |C|, alpha=0.05", 192.4, cr["0.05"]["avg_set_size"], 1)
    check(w, "row+col mean |C|, alpha=0.1", 88.7, cr["0.1"]["avg_set_size"], 1)
    w = "Sect. 4.1"
    check(w, "params, per-position (M)", 3.88, base["num_params"] / 1e6, 2)
    check(w, "params, row+col (M)", 3.73, rowc["num_params"] / 1e6, 2)
    check(w, "row+col top-1", 0.312, rowc["test_top1"], 3)
    check(w, "set-size increase at 90% (%)", 68,
          100 * (cr["0.1"]["avg_set_size"] / cb["0.1"]["avg_set_size"] - 1), 0)
    check("Sect. 7", "top-1 difference", 0.007,
          base["test_top1"] - rowc["test_top1"], 3)


def fish_simulation():
    insp = load(FISH, "inspection.json")
    w = "Sect. 5.1"
    check(w, "median kick interval (s)", 0.434, insp["median_dt_s"], 3)
    check(w, "experiments with identical start", 1, insp["seed_sharing"]["identical"], 0)
    check(w, "experiments with partial overlap", 13, insp["seed_sharing"]["partial"], 0)

    b = load(FISH, "baselines.json")["results"]
    ev = {T: load(FISH, "gru_T%d" % T, "evaluation.json")["top-1 accuracy"][0]
          for T in (16, 32, 64, 128, 256, 512)}
    tr = load(FISH, "transformer_inv_T64", "evaluation.json")
    w = "Fig. 1 / Sect. 5.2"
    check(w, "RF single-agent", 0.352, b["T64"]["rf_naive"]["acc"][0], 3)
    check(w, "RF + relational", 0.717, b["T64"]["rf"]["acc"][0], 3)
    check(w, "GRU", 0.929, ev[64], 3)
    check(w, "transformer", 0.927, tr["top-1 accuracy"][0], 3)
    check(w, "relational gain", 0.365, b["T64"]["rf"]["acc"][0] - b["T64"]["rf_naive"]["acc"][0], 3)
    check(w, "time-course gain", 0.212, ev[64] - b["T64"]["rf"]["acc"][0], 3)

    w = "Sect. 3.2"
    raw40 = load(FISH, "transformer_T64", "evaluation.json")["top-1 accuracy"][0]
    check(w, "raw transformer, 40 epochs", 0.825, raw40, 3)
    for k, rep in ((0, 0.939), (1, 0.948)):
        z = np.load(os.path.join(FISH, "transformer_raw120_T64", "probs_fold%d.npz" % k))
        acc = float((z["test_probs"].argmax(1) == z["test_labels"]).mean())
        check(w, "raw transformer, 120 epochs, fold %d" % k, rep, acc, 3)

    cap = load(FISH, "capacity_stats.json")
    pt, bs = cap["point"], cap["bootstrap"]["exp_id"]
    w = "Table 2 / Sect. 5.2"
    check(w, "params, 2 layers d=48", 148091, cap["tags"]["small"]["n_params"], 0)
    check(w, "params, 4 layers d=40", 182771, cap["tags"]["depth"]["n_params"], 0)
    check(w, "params, 4 layers d=128", 1812619, cap["tags"]["large"]["n_params"], 0)
    check(w, "acc, 2 layers d=48", 0.803, pt["acc_small"], 3)
    check(w, "acc, 4 layers d=40", 0.785, pt["acc_depth"], 3)
    check(w, "acc, 4 layers d=128", 0.927, pt["acc_large"], 3)
    check(w, "acc, GRU", 0.929, pt["acc_gru"], 3)
    check(w, "GRU - transformer", 0.0015, pt["d_gru_minus_large"], 4)
    check(w, "GRU - transformer CI lo", -0.003, bs["d_gru_minus_large"]["ci_lo"], 3)
    check(w, "GRU - transformer CI hi", 0.006, bs["d_gru_minus_large"]["ci_hi"], 3)
    check(w, "tenfold params gain", 0.142, -pt["d_depth_minus_large"], 3)
    check(w, "tenfold gain CI lo", 0.129, -bs["d_depth_minus_large"]["ci_hi"], 3)
    check(w, "tenfold gain CI hi", 0.156, -bs["d_depth_minus_large"]["ci_lo"], 3)
    check(w, "deeper - shallower", -0.018, pt["d_depth_minus_small"], 3)
    check(w, "deeper CI lo", -0.028, bs["d_depth_minus_small"]["ci_lo"], 3)
    check(w, "deeper CI hi", -0.008, bs["d_depth_minus_small"]["ci_hi"], 3)

    w = "Table 3"
    table3 = {16: (0.254, 0.549, 0.771, 0.949, 3.28, 1.4), 32: (0.291, 0.624, 0.869, 0.957, 1.86, 44),
              64: (0.352, 0.717, 0.929, 0.968, 1.22, 79), 128: (0.434, 0.809, 0.967, 0.983, 1.07, 93),
              256: (0.533, 0.885, 0.984, 0.989, 1.02, 98), 512: (0.661, 0.962, 0.970, 0.986, 1.06, 94)}
    for T, (rfn, rf, gru, cov, size, single) in table3.items():
        c = load(FISH, "gru_T%d" % T, "conformal.json")["alphas"]["0.05"]
        check(w, "T=%d single RF" % T, rfn, b["T%d" % T]["rf_naive"]["acc"][0], 3)
        check(w, "T=%d relat. RF" % T, rf, b["T%d" % T]["rf"]["acc"][0], 3)
        check(w, "T=%d GRU" % T, gru, ev[T], 3)
        check(w, "T=%d coverage" % T, cov, c["empirical_coverage"], 3)
        check(w, "T=%d mean |C|" % T, size, c["mean_set_size"], 2)
        check(w, "T=%d singletons (%%)" % T, single, 100 * c["singleton_rate"], 1 if T == 16 else 0)
    c16 = load(FISH, "gru_T16", "conformal.json")["alphas"]["0.05"]
    check(w, "T=16 coverage CI lo", 0.944, c16["coverage_ci95"][0], 3)
    check(w, "T=16 coverage CI hi", 0.955, c16["coverage_ci95"][1], 3)
    w = "Sect. 5.3"
    check(w, "temporal gain T=16", 0.222, ev[16] - b["T16"]["rf"]["acc"][0], 3)
    check(w, "temporal gain T=512", 0.008, ev[512] - b["T512"]["rf"]["acc"][0], 3)

    w = "Fig. 2"
    for a, rep in (("0.05", 3.28), ("0.10", 1.97), ("0.20", 1.39)):
        check(w, "T=16 mean |C| at %s" % a, rep,
              load(FISH, "gru_T16", "conformal.json")["alphas"][a]["mean_set_size"], 2)
    for T, rep in ((16, 33), (32, 71), (64, 91), (128, 96), (256, 99), (512, 98)):
        check(w, "T=%d singletons at 90%% (%%)" % T, rep,
              100 * load(FISH, "gru_T%d" % T, "conformal.json")["alphas"]["0.10"]["singleton_rate"], 0)

    w = "Sect. 5.3 (field)"
    ct = load(FISH, "transformer_inv_T64", "conformal.json")["alphas"]["0.05"]
    cf = load(FISH, "field_T64", "conformal.json")["alphas"]["0.05"]
    pf, yf = pooled("field_T64")
    check(w, "transformer 95% |C|", 1.28, ct["mean_set_size"], 2)
    check(w, "transformer singletons (%)", 73, 100 * ct["singleton_rate"], 0)
    check(w, "field 95% |C|", 2.65, cf["mean_set_size"], 2)
    check(w, "field singletons (%)", 0, 100 * cf["singleton_rate"], 0)
    check(w, "field accuracy", 0.690, float((pf.argmax(1) == yf).mean()), 3)
    check(w, "field coverage", 0.954, cf["empirical_coverage"], 3)

    w = "Sect. 5.4 / Fig. 3"
    check(w, "exact rule", 0.927, tr["top-1 accuracy"][0], 3)
    check(w, "strategy family", 0.939, tr["family accuracy"][0], 3)
    check(w, "neighbor count k", 0.959, tr["k accuracy"][0], 3)
    p, y = pooled("transformer_inv_T64")
    yhat = p.argmax(1)
    err = yhat != y
    same_k = K_OF[yhat] == K_OF[y]
    same_f = FAMILY_OF[yhat] == FAMILY_OF[y]
    check(w, "errors: right k, wrong family (%)", 43, 100 * (err & same_k & ~same_f).sum() / err.sum(), 0)
    check(w, "errors: right family, wrong k (%)", 16, 100 * (err & same_f & ~same_k).sum() / err.sum(), 0)
    for r, rep in ((9, 0.82), (6, 0.87), (8, 0.88), (10, 0.90), (0, 1.00)):
        m = y == r
        check(w, "recall %s" % RULES[r], rep, float((yhat[m] == r).mean()), 2)
    geo = {row["rule_id"]: row for row in load(FISH, "robot_geometry.json")["rows"]}
    for r, rep in ((1, 0.280), (2, 0.215), (3, 0.193)):
        check(w, "NN distance %s" % RULES[r], rep, geo[r]["sim"], 3)


def fish_robots():
    cmp_ = load(FISH, "robot_transfer_comparison.json")
    geo_all = load(FISH, "robot_geometry.json")
    geo = {row["rule_id"]: row for row in geo_all["rows"]}
    tr = cmp_["models"]["transformer"]["summary"]
    gr = cmp_["models"]["gru"]["summary"]

    w = "Sect. 6"
    check(w, "robot windows", 1961, cmp_["n_windows"], 0)
    check(w, "robot tau (s)", 0.920, cmp_["tau_robot"], 3)
    check(w, "tau ratio", 2.1, cmp_["tau_ratio"], 1)
    check(w, "top-3 gain from time units", 0.088,
          geo_all["transfer_tests"]["unit_conversion_gain_transformer"]["top3"], 3)
    sweep = [load(FISH, "robot_tau_sweep", "15_%s.json" % v)["models"]
             for v in ("legacy", "pooled", "filemean")]
    for m in ("transformer", "gru"):
        vals = [s[m]["summary"]["robot_top1"]["mean"] for s in sweep]
        check_below(w, "%s top-1 spread, tau 0.84-0.98 s" % m,
                    0.002, max(vals) - min(vals), 3)

    w = "Sect. 6.1"
    check(w, "transformer robot top-1", 0.365, tr["robot_top1"]["mean"], 3)
    check(w, "transformer robot top-1 std", 0.005, tr["robot_top1"]["std"], 3)
    check(w, "transformer sim. accuracy", 0.927, tr["sim_test_acc"]["mean"], 3)
    recalls = [tr["per_rule_recall"][r]["mean"] for r in tr["per_rule_recall"]]
    check(w, "mean per-rule recall", 0.24, float(np.mean(recalls)), 2)
    check(w, "GRU robot top-1", 0.277, gr["robot_top1"]["mean"], 3)
    check(w, "GRU robot top-1 std", 0.050, gr["robot_top1"]["std"], 3)
    t = geo_all["transfer_tests"]
    check(w, "paired t-test top-1, p", 0.025, t["paired_t_top1"]["p"], 3)
    check(w, "GRU top-3", 0.566, gr["robot_top3"]["mean"], 3)
    check(w, "transformer top-3", 0.574, tr["robot_top3"]["mean"], 3)
    check(w, "paired t-test top-3, p", 0.81, t["paired_t_top3"]["p"], 2)
    check(w, "Spearman rho", -0.72, geo_all["spearman_rho"], 2)

    w = "Table 4"
    table4 = {0: (0.549, 0.509, 0.93, 0.89, 0.50), 1: (0.280, 0.406, 1.45, 0.54, 0.63),
              4: (0.333, 0.360, 1.08, 0.25, 0.25), 7: (0.273, 0.283, 1.04, 0.34, 0.72),
              2: (0.215, 0.338, 1.57, 0.15, 0.07), 5: (0.237, 0.297, 1.26, 0.04, 0.02),
              8: (0.209, 0.307, 1.47, 0.16, 0.08), 3: (0.193, 0.296, 1.54, 0.03, 0.00),
              6: (0.205, 0.285, 1.39, 0.00, 0.00), 10: (0.182, 0.313, 1.71, 0.00, 0.00)}
    for r, (s, rb, ratio, rt, rg) in table4.items():
        name = RULES[r]
        check(w, "%s sim. distance" % name, s, geo[r]["sim"], 3)
        check(w, "%s robot distance" % name, rb, geo[r]["robots"], 3)
        check(w, "%s ratio" % name, ratio, geo[r]["ratio"], 2)
        check(w, "%s recall, transformer" % name, rt, tr["per_rule_recall"][name]["mean"], 2)
        check(w, "%s recall, GRU" % name, rg, gr["per_rule_recall"][name]["mean"], 2)

    w = "Table 5"
    table5 = {"0.05": (0.470, 1.7, 0.952, 10.2, 0.461, 0.947),
              "0.10": (0.429, 1.4, 0.913, 9.3, 0.404, 0.897),
              "0.20": (0.402, 1.2, 0.818, 7.3, 0.349, 0.797)}
    for a, (sc, ss, rc, rs, gsc, grc) in table5.items():
        ct, cg = tr["conformal"][a], gr["conformal"][a]
        check(w, "transformer sim-cal coverage, alpha=%s" % a, sc, ct["sim_calibrated"]["coverage"]["mean"], 3)
        check(w, "transformer sim-cal |C|, alpha=%s" % a, ss, ct["sim_calibrated"]["mean_set_size"]["mean"], 1)
        check(w, "transformer robot-cal coverage, alpha=%s" % a, rc, ct["robot_calibrated"]["coverage"]["mean"], 3)
        check(w, "transformer robot-cal |C|, alpha=%s" % a, rs, ct["robot_calibrated"]["mean_set_size"]["mean"], 1)
        check(w, "GRU sim-cal coverage, alpha=%s" % a, gsc, cg["sim_calibrated"]["coverage"]["mean"], 3)
        check(w, "GRU robot-cal coverage, alpha=%s" % a, grc, cg["robot_calibrated"]["coverage"]["mean"], 3)


def main():
    schelling()
    fish_simulation()
    fish_robots()
    width = max(len(r[1]) for r in ROWS)
    last = None
    for where, what, rep, val, digits, status in ROWS:
        if where != last:
            print("\n%s" % where)
            last = where
        print("  %-*s  paper %-10s computed %-12s %s"
              % (width, what, ("%.*f" % (digits, rep)), ("%.*f" % (digits + 1, val)), status))
    n_bad = sum(r[5] == "MISMATCH" for r in ROWS)
    n_near = sum(r[5] == "~" for r in ROWS)
    print("\n%d numbers checked: %d match, %d within one unit of the last digit, %d mismatch"
          % (len(ROWS), len(ROWS) - n_bad - n_near, n_near, n_bad))
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
