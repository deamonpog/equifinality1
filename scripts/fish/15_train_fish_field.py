"""Train the field Transformer on rendered density/momentum grids.

The field-ablation counterpart of scripts/fish/09_train_fish_transformer.py.
Everything about the protocol is held fixed against that script -- the same
runs.npz, the same windows_T64.npz fold indices, the same grouped 5-fold CV,
temperature on validation only, early stopping, and the probs_fold{k}.npz
output contract -- so steps 10 (RAPS) and 11 (evaluation) consume the output
with zero changes. The ONLY thing that differs is the observation: each raw
[T, 5, 20] window is rendered into a [T, G, G, 3] field before the model sees
it, discarding per-agent identity and sub-cell geometry.

Because the field is a deterministic, provably lossy function of the
trajectory, this run measures the identifiability ceiling of the coarser
observation, not of a worse model: even at matched parameter budget and matched
convergence, no model on the field can recover what the render threw away.

Usage:
    python scripts/fish/15_train_fish_field.py --config configs/fish_field_T64.yaml
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.getcwd())

from genmod.fish.features import FEATURE_NAMES
from genmod.fish.field import FieldStandardizer, augment_positions, render_field
from genmod.fish.gpu_transform import cut_batch
from genmod.models.field_transformer import FieldTransformer

DATA_DIR = os.path.join("data", "fish")
N_RULES = 11
DT_COL = FEATURE_NAMES.index("dt")
POS_COLS = slice(0, 2)      # x, y
VEL_COLS = slice(2, 4)      # vx, vy


class FieldSplit:
    """One partition's windows, cut and rendered to fields on demand.

    Mirrors the Split class in step 09: batches are gathered from the shared
    feature tensor by fancy indexing in the main process, then rendered and
    standardized on the GPU over the whole batch. Only the train split carries
    augmentation, and augmentation acts in coordinate space before splatting.
    """

    def __init__(self, feats, offsets, index, labels, T, device, cfg, aug):
        self.feats, self.offsets, self.index = feats, offsets, index
        self.labels = labels[index[:, 0]]
        self.runs = index[:, 0]
        self.T, self.device, self.cfg, self.aug = T, device, cfg, aug
        self.batch_size = cfg["batch_size"]
        self.G = cfg["grid"]
        self.sigma = cfg["sigma"]
        self.splat = cfg["splat"]
        # Set after the training statistics have been fitted.
        self.standardizer = None
        self.dt_mean = 0.0
        self.dt_std = 1.0

    def __len__(self):
        return self.index.shape[0]

    def _order(self, shuffle, generator):
        n = len(self)
        if shuffle:
            return torch.randperm(n, generator=generator).numpy()
        return np.arange(n)

    def _render(self, rows, do_aug):
        raw = cut_batch(self.feats, self.offsets, self.index, rows, self.T)
        raw = torch.from_numpy(np.ascontiguousarray(raw)).to(
            self.device, non_blocking=True).float()
        pos = raw[..., POS_COLS]
        vel = raw[..., VEL_COLS]
        dt = raw[:, :, 0, DT_COL]                 # [B, T]; dt is equal across agents
        if do_aug and self.aug is not None:
            pos, vel = augment_positions(pos, vel, self.aug["rotate"],
                                         self.aug["reflect"])
        field = render_field(pos, vel, self.G, self.sigma, self.splat)
        return field, dt

    def batches(self, shuffle=False, generator=None, drop_last=False,
                standardize=True, augment=False):
        order = self._order(shuffle, generator)
        for i in range(0, len(order), self.batch_size):
            rows = order[i:i + self.batch_size]
            if drop_last and rows.size < self.batch_size:
                break
            field, dt = self._render(rows, do_aug=augment)
            if standardize and self.standardizer is not None:
                field = self.standardizer.apply(field)
                dt = (dt - self.dt_mean) / self.dt_std
            y = torch.from_numpy(self.labels[rows].astype(np.int64)).to(self.device)
            yield field, dt, y


def build_splits(cfg, feats, offsets, labels, wins, fold, device):
    T = cfg["window"]
    aug = None
    if cfg["augment"]["enabled"]:
        aug = {"rotate": cfg["augment"]["rotate"],
               "reflect": cfg["augment"]["reflect"]}
    splits = {}
    for name in ("train", "val", "calib", "test"):
        splits[name] = FieldSplit(
            feats, offsets, wins["fold%d_%s" % (fold, name)], labels, T, device,
            cfg, aug=aug if name == "train" else None)
    return splits


def fit_field_stats(cfg, train_split):
    """Per-channel field stats and dt stats, over TRAIN windows only.

    One un-augmented pass over the fold's training windows. Augmentation is a
    rotation/reflection, both arena symmetries, so the channel distribution is
    the same with or without it; fitting on the un-augmented pass keeps the
    stats reproducible.
    """
    std = FieldStandardizer(clip=cfg.get("clip_std", 6.0))
    n = 0
    s = 0.0
    s2 = 0.0
    for field, dt, _ in train_split.batches(shuffle=False, standardize=False,
                                            augment=False):
        std.update(field)
        d = dt.reshape(-1).double()
        n += d.numel()
        s += float(d.sum())
        s2 += float((d * d).sum())
    std.finalize()
    dt_mean = s / n
    dt_std = math.sqrt(max(s2 / n - dt_mean * dt_mean, 1e-12))
    return std, dt_mean, dt_std


@torch.no_grad()
def predict(model, split):
    """Logits and labels for a whole partition (no augmentation)."""
    model.eval()
    L, Y = [], []
    for field, dt, y in split.batches(shuffle=False, standardize=True,
                                      augment=False):
        L.append(model(field, dt).float().cpu())
        Y.append(y.cpu())
    return torch.cat(L), torch.cat(Y)


def fit_temperature(logits, labels):
    """One scalar temperature, by LBFGS on validation NLL (as in step 09)."""
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
    splits = build_splits(cfg, feats, offsets, labels, wins, fold, device)

    # Fit field/dt standardization on TRAIN only, then share it with all splits.
    std, dt_mean, dt_std = fit_field_stats(cfg, splits["train"])
    for sp in splits.values():
        sp.standardizer = std
        sp.dt_mean = dt_mean
        sp.dt_std = dt_std

    model = FieldTransformer(
        in_channels=3,
        patch=cfg["patch"],
        grid=cfg["grid"],
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

    print("  fold %d: %d params, %d train windows, G=%d patch=%d"
          % (fold, n_params, len(splits["train"]), cfg["grid"], cfg["patch"]))

    history = []
    best_acc, best_state, bad_epochs = -1.0, None, 0
    for epoch in range(cfg["epochs"]):
        model.train()
        t0, tot, seen = time.time(), 0.0, 0
        for field, dt, y in splits["train"].batches(
                shuffle=True, generator=gen, drop_last=True,
                standardize=True, augment=True):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                loss = lossfn(model(field, dt), y)
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

    # Temperature on VALIDATION, never on calibration.
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

    np.savez_compressed(os.path.join(out_dir, "probs_fold%d.npz" % fold), **out)
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
    print("Config %s   device %s   T=%d   G=%d   splat=%s   augment=%s"
          % (args.config, device, cfg["window"], cfg["grid"], cfg["splat"],
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

    with open(os.path.join(out_dir, "training_log.csv"), "w") as fh:
        fh.write("fold,epoch,train_loss,val_acc\n")
        for fold, epoch, train_loss, val_acc in log_rows:
            fh.write("%d,%d,%.6f,%.6f\n" % (fold, epoch, train_loss, val_acc))

    n_params = next(iter(n_params_by_fold.values())) if n_params_by_fold else None
    summary = {
        "tag": tag, "config": args.config, "n_params": n_params,
        "n_params_by_fold": n_params_by_fold,
        "test_acc_mean": float(np.mean(summaries)),
        "test_acc_std": float(np.std(summaries)),
        "epochs_budget": cfg["epochs"],
        "representation": "field",
        "grid": cfg["grid"], "patch": cfg["patch"], "splat": cfg["splat"],
    }
    with open(os.path.join(out_dir, "run_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    print("params=%s  wrote %s" % (n_params, out_dir))


if __name__ == "__main__":
    main()
