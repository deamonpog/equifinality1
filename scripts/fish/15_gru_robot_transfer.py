"""Robot transfer, transformer vs GRU, across all five folds (Sect. 6).

Takes the simulation-trained models, without retraining, and applies them to
the robot trajectories of the Lei et al. (2020) release: five cuboid robots,
each programmed with one of the simulated neighbor-selection rules.

  * The TRANSFORMER is the per-agent invariant-feature model
    (results/fish/transformer_inv_T64/model_fold{k}.pt), one checkpoint per
    fold.
  * The GRU consumes the 13 per-event GROUP statistics
    (per_event_group_features), which average the five agents away at every
    event. It saves no weights, so each fold is retrained here on the same
    simulated training split (~30 s per fold on a GPU).

If both representations fail in the same way, the limit lies in the relation
between the simulated rule library and the robot substrate, not in either
classifier.

For each model and fold the script reports robot top-1/top-3, family and k
accuracy and per-rule recall, with time both in seconds and in each domain's
median kick interval, and conformal coverage and set size with (a) the
simulation calibration set and (b) a calibration set of held-out robot
experiments.

Symmetry matters: we run the transformer and the GRU through the identical
robot pipeline -- same windows, same nondimensionalization, same conformal
split -- so the only difference is the model and its representation.

Scope: accuracy (top-1/top-3, family/k, per-rule recall) AND conformal parity
(sim-calibrated vs robot-recalibrated coverage and set size). Everything runs
on one GPU in well under fifteen minutes.

Usage:
    python scripts/fish/15_gru_robot_transfer.py --window 64
    python scripts/fish/15_gru_robot_transfer.py --window 64 --folds 0 1 2 3 4
    python scripts/fish/15_gru_robot_transfer.py --tau-robot 0.840 --out x.json

By default both kick intervals (tau) are computed from the data
(robots.robot_tau / robots.agent_tau); --tau-robot / --tau-agent override them.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.getcwd())

from genmod.evaluation.conformal import (compute_nonconformity_scores,
                                         conformal_calibrate, conformal_predict)
from genmod.fish.dataset import GroupSequenceDataset
from genmod.fish.features import (FEATURE_NAMES, INVARIANT_COLS, compute_features)
from genmod.fish.gpu_transform import GPUTransform
from genmod.fish.robots import (ROBOT_RADIUS, agent_tau, load_robot_runs,
                                robot_tau, to_agent_units)
from genmod.fish.summary import PER_EVENT_NAMES, per_event_group_features
from genmod.models.traj_transformer import GroupGRU, TrajectoryTransformer

DATA_DIR = os.path.join("data", "fish")
TRANSFORMER_TAG = "transformer_inv_T64"
GRU_TAG = "gru_T64"
N_RULES = 11
BURN_IN = 10   # robot runs are short (140-730 events); 50 would waste a third
ALPHAS = [0.05, 0.10, 0.20]
K_REG, LAMBDA_REG = 2, 0.1   # RAPS params, as in scripts/fish/10
SEED = 20260714

# The 11-class rule label space, shared by agents and robots. Only ten of the
# eleven rules have robot files.
RULE_NAMES = ["none-0", "near-1", "near-2", "near-3", "rand-1", "rand-2",
              "rand-3", "infl-1", "infl-2", "infl-3", "all-4"]
K_OF = np.array([0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4])
FAMILY_OF = np.array([0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4])

TRAIN_SPLITS = ("train", "val", "calib", "test")


# ---------------------------------------------------------------------------
# Part A: GRU fold retrain -- lifted from 08_train_gru_baseline.py so the
# retrained GRU is identical to the paper's baseline. Unlike script 08 we keep
# the (model, mu, sd, temperature) tuple in memory for the transfer.
# ---------------------------------------------------------------------------
def build_sequences(feats, offsets, feature_names, index, T):
    X = np.empty((index.shape[0], T, len(PER_EVENT_NAMES)), dtype=np.float32)
    for n, (run_idx, start) in enumerate(index):
        o = offsets[run_idx]
        X[n] = per_event_group_features(
            feats[o + start:o + start + T], feature_names)
    return X


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    L, Y = [], []
    for x, y in loader:
        L.append(model(x.to(device)).float().cpu())
        Y.append(y)
    return torch.cat(L), torch.cat(Y)


def fit_temperature(logits, labels):
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
    nll = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().detach())


def train_gru_fold(args, fold, device):
    """Retrain GRU fold `fold`; return (model, mu, sd, temp, sim_test_acc)."""
    T = args.window
    data = np.load(os.path.join(DATA_DIR, "runs.npz"), allow_pickle=True)
    feats, offsets = data["feats"], data["offsets"]
    labels = data["rule_id"]
    feature_names = [str(s) for s in data["feature_names"]]
    wins = np.load(os.path.join(DATA_DIR, "windows_T%d.npz" % T))

    seqs, ys, runs = {}, {}, {}
    for split in TRAIN_SPLITS:
        idx = wins["fold%d_%s" % (fold, split)]
        seqs[split] = build_sequences(feats, offsets, feature_names, idx, T)
        ys[split] = labels[idx[:, 0]]
        runs[split] = idx[:, 0]

    # Standardize on TRAIN ONLY. Keep mu/sd: robot data must use these units.
    mu = seqs["train"].reshape(-1, seqs["train"].shape[-1]).mean(0)
    sd = seqs["train"].reshape(-1, seqs["train"].shape[-1]).std(0) + 1e-6
    for split in TRAIN_SPLITS:
        seqs[split] = (seqs[split] - mu) / sd

    loaders = {
        s: DataLoader(GroupSequenceDataset(seqs[s], ys[s], runs[s]),
                      batch_size=args.batch_size, shuffle=(s == "train"))
        for s in TRAIN_SPLITS
    }

    model = GroupGRU(len(PER_EVENT_NAMES), N_RULES,
                     hidden=args.hidden, n_layers=args.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    lossfn = nn.CrossEntropyLoss(label_smoothing=0.05)

    best, best_state, bad = -1.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = lossfn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        vl, vy = predict(model, loaders["val"], device)
        va = float((vl.argmax(1) == vy).float().mean())
        if va > best:
            best, bad = va, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break

    model.load_state_dict(best_state)
    vl, vy = predict(model, loaders["val"], device)
    temp = fit_temperature(vl, vy)
    tl, ty = predict(model, loaders["test"], device)
    test_acc = float((tl.argmax(1) == ty).float().mean())
    return model, mu.astype(np.float32), sd.astype(np.float32), temp, test_acc


def load_transformer_fold(fold, T, device):
    """Load the saved transformer checkpoint; return (model, tf, temp, acc)."""
    res_dir = os.path.join("results", "fish", TRANSFORMER_TAG)
    ck = torch.load(os.path.join(res_dir, "model_fold%d.pt" % fold),
                    map_location=device, weights_only=False)
    cfg = ck["cfg"]
    model = TrajectoryTransformer(
        n_features=len(INVARIANT_COLS), n_agents=5, max_events=T,
        num_classes=N_RULES, d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"], dropout=cfg["dropout"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    temp = ck["temperature"]

    wins = np.load(os.path.join(DATA_DIR, "windows_T%d.npz" % T))
    tf = GPUTransform(wins["fold%d_median" % fold], wins["fold%d_iqr" % fold],
                      float(wins["clip"]), device)

    sim = np.load(os.path.join(res_dir, "probs_fold%d.npz" % fold))
    acc = float((sim["test_probs"].argmax(1) == sim["test_labels"]).mean())
    return model, tf, temp, acc


# ---------------------------------------------------------------------------
# Inference: same raw robot windows X [N, T, 5, 20] into each model's own
# representation. GRU -> per-event group stats; transformer -> scaled invariant
# columns. (Mirrors `infer` in scripts 08 and 14 respectively.)
# ---------------------------------------------------------------------------
def gru_infer(model, X, mu, sd, temp, device):
    G = len(PER_EVENT_NAMES)
    seq = np.empty((X.shape[0], X.shape[1], G), dtype=np.float32)
    for n in range(X.shape[0]):
        seq[n] = per_event_group_features(X[n], list(FEATURE_NAMES))
    seq = (seq - mu) / sd
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, seq.shape[0], 256):
            xb = torch.from_numpy(np.ascontiguousarray(seq[i:i + 256])).to(device)
            out.append(torch.softmax(model(xb) / temp, dim=1).cpu().numpy())
    return np.concatenate(out)


def transformer_infer(model, tf, temp, X, device):
    out = []
    with torch.no_grad():
        for i in range(0, X.shape[0], 256):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + 256])).to(device)
            xb = tf.scale(xb)[..., INVARIANT_COLS]
            out.append(torch.softmax(model(xb) / temp, dim=1).cpu().numpy())
    return np.concatenate(out)


def coverage_and_size(probs, labels, qhat):
    sets = conformal_predict(probs, qhat, method="raps",
                             k_reg=K_REG, lambda_reg=LAMBDA_REG)
    cover = np.array([y in s for s, y in zip(sets, labels)], dtype=float)
    size = np.array([len(s) for s in sets], dtype=float)
    return float(cover.mean()), float(size.mean())


def acc_block(p, y, chance):
    pr = p.argmax(1)
    return {
        "top1": float((pr == y).mean()),
        "top1_x_chance": float((pr == y).mean() / chance),
        "top3": float(np.mean([y[i] in np.argsort(-p[i])[:3]
                               for i in range(len(y))])),
        "family_acc": float((FAMILY_OF[pr] == FAMILY_OF[y]).mean()),
        "k_acc": float((K_OF[pr] == K_OF[y]).mean()),
    }


def conformal_block(sim_calib_probs, sim_calib_labels,
                    probs_nd, y, cal_m, test_m):
    """Sim-calibrated (the trap) vs robot-recalibrated coverage/size."""
    sim_scores = compute_nonconformity_scores(
        sim_calib_probs, sim_calib_labels, method="raps",
        k_reg=K_REG, lambda_reg=LAMBDA_REG)
    rob_scores = compute_nonconformity_scores(
        probs_nd[cal_m], y[cal_m], method="raps",
        k_reg=K_REG, lambda_reg=LAMBDA_REG)
    block = {}
    for a in ALPHAS:
        cov_s, sz_s = coverage_and_size(
            probs_nd[test_m], y[test_m], conformal_calibrate(sim_scores, alpha=a))
        cov_r, sz_r = coverage_and_size(
            probs_nd[test_m], y[test_m], conformal_calibrate(rob_scores, alpha=a))
        block["%.2f" % a] = {
            "target": 1 - a,
            "sim_calibrated": {"coverage": cov_s, "mean_set_size": sz_s},
            "robot_calibrated": {"coverage": cov_r, "mean_set_size": sz_r},
        }
    return block


def mean_std(vals):
    v = np.array(vals, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--tau-robot", type=float, default=None,
                    help="robot kick interval, s (default: robot_tau())")
    ap.add_argument("--tau-agent", type=float, default=None,
                    help="agent kick interval, s (default: agent_tau())")
    ap.add_argument("--out", default=os.path.join(
        "results", "fish", "robot_transfer_comparison.json"))
    args = ap.parse_args()
    tau_robot = args.tau_robot if args.tau_robot is not None else robot_tau()
    tau_agent = args.tau_agent if args.tau_agent is not None else agent_tau()
    print("tau_robot %.3f s  tau_agent %.3f s  ratio %.3f"
          % (tau_robot, tau_agent, tau_robot / tau_agent))

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = args.window
    print("device %s, folds %s\n" % (device, args.folds))

    # ---- robot data: cut ONCE, shared by every model and fold -----------
    print("Loading robot runs ...")
    runs = load_robot_runs()
    rule_ids = sorted({r.rule_id for r in runs})
    missing = [RULE_NAMES[i] for i in range(N_RULES) if i not in rule_ids]
    print("  %d runs, %d rules present (of %d)  [missing: %s]"
          % (len(runs), len(rule_ids), N_RULES, ", ".join(missing) or "none"))

    feats, meta = [], []
    for run in runs:
        f, _ = compute_features(run.time, run.pos,
                                radius=ROBOT_RADIUS, burn_in=BURN_IN)
        if f.shape[0] < T:
            continue
        for s in range(0, f.shape[0] - T + 1, T):   # non-overlapping only
            feats.append(f[s:s + T])
            meta.append((run.rule_id, run.exp_id, len(meta)))
    X = np.stack(feats)                                   # [N, T, 5, 20]
    X_nd = to_agent_units(X, tau_robot, tau_agent)       # per-kick units
    y = np.array([m[0] for m in meta])
    grp = np.array([m[1] for m in meta])
    chance = 1.0 / len(rule_ids)
    print("  %d non-overlapping windows of T=%d (%.0f s each), chance %.4f"
          % (X.shape[0], T, T * tau_robot, chance))

    # Robot conformal split: by EXPERIMENT, shared across models/folds so the
    # comparison is apples-to-apples.
    rng = np.random.default_rng(SEED)
    exps = np.unique(grp)
    perm = rng.permutation(exps)
    cal_exps = set(perm[:len(perm) // 2].tolist())
    cal_m = np.array([g in cal_exps for g in grp])
    test_m = ~cal_m
    print("  conformal: %d calib / %d eval windows (%d/%d experiments)\n"
          % (cal_m.sum(), test_m.sum(), len(cal_exps), len(exps) - len(cal_exps)))

    models = [
        ("transformer", TRANSFORMER_TAG,
         lambda fold: _run_transformer(fold, T, device, X, X_nd)),
        ("gru", GRU_TAG,
         lambda fold: _run_gru(args, fold, device, X, X_nd)),
    ]

    results = {"window": T, "folds": args.folds, "n_windows": int(X.shape[0]),
               "tau_robot": tau_robot, "tau_agent": tau_agent,
               "tau_ratio": tau_robot / tau_agent,
               "rules_present": rule_ids, "models": {}}

    for name, tag, runner in models:
        print("=" * 70)
        print("MODEL: %s" % name)
        print("=" * 70)
        per_fold = []
        for fold in args.folds:
            t0 = time.time()
            probs_raw, probs_nd, sim_test_acc = runner(fold)
            a_raw = acc_block(probs_raw, y, chance)
            a_nd = acc_block(probs_nd, y, chance)

            sim = np.load(os.path.join("results", "fish", tag,
                                       "probs_fold%d.npz" % fold))
            conf = conformal_block(sim["calib_probs"], sim["calib_labels"],
                                   probs_nd, y, cal_m, test_m)

            pred = probs_nd.argmax(1)
            per_rule = {RULE_NAMES[c]: float((pred[y == c] == c).mean())
                        for c in rule_ids}

            rec = {"fold": fold, "sim_test_acc": sim_test_acc,
                   "robot_seconds": a_raw, "robot_nondim": a_nd,
                   "unit_conversion_gain": a_nd["top1"] - a_raw["top1"],
                   "per_rule_recall": per_rule, "conformal": conf}
            per_fold.append(rec)
            print("  fold %d  sim_test %.4f  robot top1 %.4f (%.1fx)  top3 %.4f"
                  "  fam %.4f  k %.4f  [%.0fs]"
                  % (fold, sim_test_acc, a_nd["top1"], a_nd["top1_x_chance"],
                     a_nd["top3"], a_nd["family_acc"], a_nd["k_acc"],
                     time.time() - t0))

        summary = {
            "sim_test_acc": mean_std([r["sim_test_acc"] for r in per_fold]),
            "robot_top1": mean_std([r["robot_nondim"]["top1"] for r in per_fold]),
            "robot_top3": mean_std([r["robot_nondim"]["top3"] for r in per_fold]),
            "robot_family_acc": mean_std([r["robot_nondim"]["family_acc"]
                                          for r in per_fold]),
            "robot_k_acc": mean_std([r["robot_nondim"]["k_acc"]
                                     for r in per_fold]),
            "per_rule_recall": {
                RULE_NAMES[c]: mean_std([r["per_rule_recall"][RULE_NAMES[c]]
                                         for r in per_fold]) for c in rule_ids},
            "conformal": {
                "%.2f" % a: {
                    which: {
                        "coverage": mean_std([r["conformal"]["%.2f" % a][which]
                                              ["coverage"] for r in per_fold]),
                        "mean_set_size": mean_std(
                            [r["conformal"]["%.2f" % a][which]["mean_set_size"]
                             for r in per_fold]),
                    } for which in ("sim_calibrated", "robot_calibrated")
                } for a in ALPHAS},
        }
        results["models"][name] = {"per_fold": per_fold, "summary": summary}
        s = summary
        print("  ---- %s over %d folds ----" % (name, len(per_fold)))
        print("    sim test    %.4f +- %.4f" % (s["sim_test_acc"]["mean"],
                                                s["sim_test_acc"]["std"]))
        print("    robot top1  %.4f +- %.4f  (%.1fx chance)"
              % (s["robot_top1"]["mean"], s["robot_top1"]["std"],
                 s["robot_top1"]["mean"] / chance))
        print("    robot top3  %.4f +- %.4f\n" % (s["robot_top3"]["mean"],
                                                  s["robot_top3"]["std"]))

    # ---- side-by-side summary ------------------------------------------
    print("=" * 70)
    print("SUMMARY  (robot, nondimensionalized, mean +- std over %d folds)"
          % len(args.folds))
    print("=" * 70)
    print("  %-12s %-18s %-18s" % ("", "transformer", "gru"))
    tr, gr = results["models"]["transformer"]["summary"], \
        results["models"]["gru"]["summary"]
    for key, lab in [("sim_test_acc", "sim test"), ("robot_top1", "robot top1"),
                     ("robot_top3", "robot top3"),
                     ("robot_family_acc", "robot family"),
                     ("robot_k_acc", "robot k")]:
        print("  %-12s %.4f +- %.4f    %.4f +- %.4f"
              % (lab, tr[key]["mean"], tr[key]["std"],
                 gr[key]["mean"], gr[key]["std"]))

    print("\n  per-rule recall (robot, mean over folds):")
    print("    %-8s %-6s %-16s %-16s" % ("rule", "k", "transformer", "gru"))
    for c in rule_ids:
        rn = RULE_NAMES[c]
        print("    %-8s %-6d %.3f +- %.3f    %.3f +- %.3f"
              % (rn, K_OF[c], tr["per_rule_recall"][rn]["mean"],
                 tr["per_rule_recall"][rn]["std"],
                 gr["per_rule_recall"][rn]["mean"],
                 gr["per_rule_recall"][rn]["std"]))

    print("\n  conformal coverage @ target (mean over folds):")
    print("    %-7s %-9s %-26s %-26s"
          % ("alpha", "target", "transformer (sim|robot)", "gru (sim|robot)"))
    for a in ALPHAS:
        k = "%.2f" % a
        tc, gc = tr["conformal"][k], gr["conformal"][k]
        print("    %-7.2f %-9.2f %.3f | %.3f  (|C| %4.1f|%4.1f)   %.3f | %.3f"
              " (|C| %4.1f|%4.1f)"
              % (a, 1 - a,
                 tc["sim_calibrated"]["coverage"]["mean"],
                 tc["robot_calibrated"]["coverage"]["mean"],
                 tc["sim_calibrated"]["mean_set_size"]["mean"],
                 tc["robot_calibrated"]["mean_set_size"]["mean"],
                 gc["sim_calibrated"]["coverage"]["mean"],
                 gc["robot_calibrated"]["coverage"]["mean"],
                 gc["sim_calibrated"]["mean_set_size"]["mean"],
                 gc["robot_calibrated"]["mean_set_size"]["mean"]))

    path = args.out
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nWrote %s" % path)


# thin closures so the model loop stays readable ---------------------------
def _run_transformer(fold, T, device, X, X_nd):
    model, tf, temp, sim_acc = load_transformer_fold(fold, T, device)
    return (transformer_infer(model, tf, temp, X, device),
            transformer_infer(model, tf, temp, X_nd, device), sim_acc)


def _run_gru(args, fold, device, X, X_nd):
    model, mu, sd, temp, sim_acc = train_gru_fold(args, fold, device)
    return (gru_infer(model, X, mu, sd, temp, device),
            gru_infer(model, X_nd, mu, sd, temp, device), sim_acc)


if __name__ == "__main__":
    main()
