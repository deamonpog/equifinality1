"""Train the trajectory Transformer on agent-time tokens.

One model per grouped fold. For each fold the script trains on the fold's
training experiments, early-stops on validation, fits a temperature on
validation, and then saves the SOFTMAX PROBABILITIES for the calibration and
test splits. Steps 10 and 11 consume those probability files, so the conformal
analysis and the evaluation never re-run training and can be iterated cheaply.

The calibration split is touched only to produce probabilities: nothing about
the model, the early stopping, or the temperature is chosen using it. That is
what keeps the RAPS coverage guarantee honest.

Usage:
    python scripts/fish/09_train_fish_transformer.py --config configs/fish_transformer.yaml
    python scripts/fish/09_train_fish_transformer.py --config configs/fish_smoke.yaml
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.getcwd())

from genmod.fish.features import FEATURE_NAMES, INVARIANT_COLS, N_BASIC
from genmod.fish.gpu_transform import GPUTransform, cut_batch
from genmod.models.traj_transformer import TrajectoryTransformer

DATA_DIR = os.path.join("data", "fish")
N_RULES = 11


class Split:
    """One partition's windows, cut and transformed on demand.

    Deliberately not a torch DataLoader. Windows overlap heavily, so batches
    are gathered from the shared feature tensor by fancy indexing in the main
    process; augmentation and scaling then happen on the GPU over the whole
    batch. Worker processes would only re-copy the 283 MB tensor every epoch.
    """

    def __init__(self, feats, offsets, index, labels, T, tf, cols,
                 batch_size, aug=None):
        self.feats, self.offsets, self.index = feats, offsets, index
        self.labels = labels[index[:, 0]]
        self.runs = index[:, 0]
        self.T, self.tf, self.cols = T, tf, cols
        self.batch_size = batch_size
        self.aug = aug

    def __len__(self):
        return self.index.shape[0]

    def batches(self, shuffle, generator=None, drop_last=False):
        n = len(self)
        order = (torch.randperm(n, generator=generator).numpy() if shuffle
                 else np.arange(n))
        for i in range(0, n, self.batch_size):
            rows = order[i:i + self.batch_size]
            if drop_last and rows.size < self.batch_size:
                break
            raw = cut_batch(self.feats, self.offsets, self.index, rows, self.T)
            x = torch.from_numpy(np.ascontiguousarray(raw)).to(
                self.tf.device, non_blocking=True)
            if self.aug is not None:
                x = self.tf.augment(x, **self.aug)
            x = self.tf.scale(x)
            if self.cols is not None:
                x = x[..., self.cols]
            y = torch.from_numpy(self.labels[rows].astype(np.int64)).to(
                self.tf.device, non_blocking=True)
            yield x, y


def build_splits(cfg, feats, offsets, labels, wins, fold, device):
    T = cfg["window"]
    tf = GPUTransform(wins["fold%d_median" % fold], wins["fold%d_iqr" % fold],
                      float(wins["clip"]), device)

    # all       the 20 raw per-agent features, including absolute coordinates
    # basic     the 11 single-agent kinematics only (no neighbor information)
    # invariant the 12 rotation-invariant features: same invariance the GRU
    #           baseline gets for free, but still per-agent
    cols = {"all": None,
            "basic": list(range(N_BASIC)),
            "invariant": INVARIANT_COLS}[cfg["features"]]

    aug = None
    if cfg["augment"]["enabled"]:
        aug = {"rotate": cfg["augment"]["rotate"],
               "reflect": cfg["augment"]["reflect"],
               "permute": cfg["augment"]["permute"]}
        # Rotation augmentation on an already rotation-invariant feature set is
        # a no-op that still costs a batch of GPU kernels. Skip it, and say so.
        if cfg["features"] == "invariant" and aug["rotate"]:
            print("  note: features=invariant, so rotation augmentation is a "
                  "no-op and is disabled")
            aug["rotate"] = False

    splits = {}
    for name in ("train", "val", "calib", "test"):
        splits[name] = Split(
            feats, offsets, wins["fold%d_%s" % (fold, name)], labels, T, tf,
            cols, cfg["batch_size"],
            # Augmentation is a TRAINING device only. Applying it at eval time
            # would change what is being measured.
            aug=aug if name == "train" else None)
    return splits, (len(cols) if cols else len(FEATURE_NAMES))


@torch.no_grad()
def predict(model, split):
    """Logits and labels for a whole partition."""
    model.eval()
    L, Y = [], []
    for x, y in split.batches(shuffle=False):
        L.append(model(x).float().cpu())
        Y.append(y.cpu())
    return torch.cat(L), torch.cat(Y)


def fit_temperature(logits, labels):
    """One scalar temperature, by LBFGS on validation NLL.

    Softens the logits without changing the argmax, so accuracy is untouched
    while calibration (and therefore the tightness of the conformal sets)
    improves.
    """
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


def run_fold(cfg, feats, offsets, labels, wins, fold, device, out_dir):
    splits, n_feat = build_splits(cfg, feats, offsets, labels, wins, fold, device)

    model = TrajectoryTransformer(
        n_features=n_feat,
        n_agents=5,
        max_events=cfg["window"],
        num_classes=N_RULES,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    steps = len(splits["train"]) // cfg["batch_size"]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg["lr"], epochs=cfg["epochs"],
        steps_per_epoch=steps, pct_start=0.1)
    lossfn = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    gen = torch.Generator().manual_seed(cfg["seed"] + fold)

    print("  fold %d: %d params, %d train windows, %d feats"
          % (fold, n_params, len(splits["train"]), n_feat))

    # Per-epoch validation curve, committed to training_log.csv. A negative
    # result about capacity is only usable if every model is trained to
    # convergence, so the plateau must be an inspectable artifact, not a claim
    # in a commit message. history rows are (epoch, train_loss, val_acc).
    history = []

    best_acc, best_state, bad_epochs = -1.0, None, 0
    for epoch in range(cfg["epochs"]):
        model.train()
        t0, tot, seen = time.time(), 0.0, 0
        for x, y in splits["train"].batches(shuffle=True, generator=gen,
                                            drop_last=True):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                loss = lossfn(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss.detach()) * y.size(0)
            seen += y.size(0)

        vl, vy = predict(model, splits["val"])
        val_acc = float((vl.argmax(1) == vy).float().mean())
        history.append((epoch, tot / seen, val_acc))
        print("    epoch %2d  loss %.4f  val_acc %.4f  (%.0fs)"
              % (epoch, tot / seen, val_acc, time.time() - t0), flush=True)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["patience"]:
                print("    early stop (no val improvement for %d epochs)"
                      % cfg["patience"])
                break

    model.load_state_dict(best_state)

    # Temperature is fitted on VALIDATION. Using calibration here would spend
    # the split twice and void the conformal guarantee.
    vl, vy = predict(model, splits["val"])
    temp = fit_temperature(vl, vy)
    print("    best val_acc %.4f, temperature %.3f" % (best_acc, temp))

    out = {"temperature": temp, "val_acc": best_acc, "n_params": n_params,
           "history": np.asarray(history, dtype=np.float64),
           "best_epoch": int(max(range(len(history)),
                                 key=lambda i: history[i][2])) if history else -1}
    for name in ("calib", "test"):
        logits, y = predict(model, splits[name])
        probs = torch.softmax(logits / temp, dim=1).numpy()
        out["%s_probs" % name] = probs
        out["%s_labels" % name] = y.numpy()
        out["%s_runs" % name] = splits[name].runs
        acc = float((probs.argmax(1) == y.numpy()).mean())
        print("    %-6s acc %.4f" % (name, acc), flush=True)

    path = os.path.join(out_dir, "probs_fold%d.npz" % fold)
    np.savez_compressed(path, **out)

    # Keep the weights: the embedding analysis and any post-hoc probe need the
    # model itself, and retraining five folds to recover it would be wasteful.
    torch.save({"state_dict": best_state, "temperature": temp, "cfg": cfg},
               os.path.join(out_dir, "model_fold%d.pt" % fold))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", default=None, help="override the output directory name")
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    tag = args.tag or cfg["tag"]
    out_dir = os.path.join("results", "fish", tag)
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["seed"])
    print("Config %s   device %s   T=%d   features=%s   augment=%s"
          % (args.config, device, cfg["window"], cfg["features"],
             cfg["augment"]["enabled"]))
    if device == "cuda":
        print("GPU: %s" % torch.cuda.get_device_name(0))

    data = np.load(os.path.join(DATA_DIR, "runs.npz"), allow_pickle=True)
    feats, offsets, labels = data["feats"], data["offsets"], data["rule_id"]
    wins = np.load(os.path.join(DATA_DIR, "windows_T%d.npz" % cfg["window"]))

    summaries, log_rows, n_params_by_fold = [], [], {}
    for fold in cfg["folds"]:
        r = run_fold(cfg, feats, offsets, labels, wins, fold, device, out_dir)
        acc = float((r["test_probs"].argmax(1) == r["test_labels"]).mean())
        summaries.append(acc)
        n_params_by_fold[fold] = int(r["n_params"])
        for epoch, train_loss, val_acc in r["history"]:
            log_rows.append((fold, int(epoch), float(train_loss), float(val_acc)))

    print("\nTest accuracy over %d folds: %.4f +- %.4f"
          % (len(summaries), float(np.mean(summaries)), float(np.std(summaries))))

    # Committed per-epoch validation curve. This is the artifact that makes a
    # convergence claim inspectable rather than asserted.
    with open(os.path.join(out_dir, "training_log.csv"), "w") as fh:
        fh.write("fold,epoch,train_loss,val_acc\n")
        for fold, epoch, train_loss, val_acc in log_rows:
            fh.write("%d,%d,%.6f,%.6f\n" % (fold, epoch, train_loss, val_acc))

    # n_params is architecture-fixed across folds; store it (and the tail-mean
    # val_acc) somewhere greppable, not only inside the per-fold npz.
    n_params = next(iter(n_params_by_fold.values())) if n_params_by_fold else None
    summary = {
        "tag": tag, "config": args.config, "n_params": n_params,
        "n_params_by_fold": n_params_by_fold,
        "test_acc_mean": float(np.mean(summaries)),
        "test_acc_std": float(np.std(summaries)),
        "epochs_budget": cfg["epochs"],
    }
    with open(os.path.join(out_dir, "run_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    print("params=%s  wrote %s" % (n_params, out_dir))


if __name__ == "__main__":
    main()
