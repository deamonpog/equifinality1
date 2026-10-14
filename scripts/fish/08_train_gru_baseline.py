"""Recurrent baseline over per-event GROUP statistics.

This is the control that decides what a Transformer win actually means.

The GRU is also a sequence model, also sees the full time course of the
window, and also has access to relational quantities (nearest-neighbor
distance, cohesion, polarization). The one thing it cannot see is WHICH AGENT
IS WHICH: at every event the five agents have already been averaged away into
a single group-state vector.

So the comparison isolates the question:

    Transformer > GRU   =>  the gain comes from preserving individual-agent
                            structure (RQ5, ablation 2)
    Transformer ~ GRU   =>  the gain came from sequence modelling alone, and
                            the agent-level representation buys nothing

Without this control, a Transformer that beats the random forest would be
ambiguous: the forest sees no time course either, so it could simply be that
any sequence model wins.

Writes probability files in the same format as step 09, so step 10 can
calibrate RAPS on the GRU too.

Usage:
    python scripts/fish/08_train_gru_baseline.py --window 64
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

from genmod.fish.dataset import GroupSequenceDataset
from genmod.fish.summary import PER_EVENT_NAMES, per_event_group_features
from genmod.models.traj_transformer import GroupGRU

DATA_DIR = os.path.join("data", "fish")
N_RULES = 11
SPLITS = ("train", "val", "calib", "test")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    T = args.window
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.path.join("results", "fish", "gru_T%d" % T)
    os.makedirs(out_dir, exist_ok=True)

    data = np.load(os.path.join(DATA_DIR, "runs.npz"), allow_pickle=True)
    feats, offsets = data["feats"], data["offsets"]
    labels = data["rule_id"]
    feature_names = [str(s) for s in data["feature_names"]]
    wins = np.load(os.path.join(DATA_DIR, "windows_T%d.npz" % T))

    print("GRU baseline, T=%d, device %s, %d group statistics per event"
          % (T, device, len(PER_EVENT_NAMES)))

    accs = []
    for fold in args.folds:
        print("\n  fold %d: building group-state sequences ..." % fold)
        seqs, ys, runs = {}, {}, {}
        for split in SPLITS:
            idx = wins["fold%d_%s" % (fold, split)]
            seqs[split] = build_sequences(feats, offsets, feature_names, idx, T)
            ys[split] = labels[idx[:, 0]]
            runs[split] = idx[:, 0]

        # Standardize on TRAIN ONLY, as everywhere else in the pipeline.
        mu = seqs["train"].reshape(-1, seqs["train"].shape[-1]).mean(0)
        sd = seqs["train"].reshape(-1, seqs["train"].shape[-1]).std(0) + 1e-6
        for split in SPLITS:
            seqs[split] = (seqs[split] - mu) / sd

        loaders = {
            s: DataLoader(GroupSequenceDataset(seqs[s], ys[s], runs[s]),
                          batch_size=args.batch_size, shuffle=(s == "train"))
            for s in SPLITS
        }

        model = GroupGRU(len(PER_EVENT_NAMES), N_RULES,
                         hidden=args.hidden, n_layers=args.layers).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        lossfn = nn.CrossEntropyLoss(label_smoothing=0.05)
        print("    %d params, %d train windows"
              % (sum(p.numel() for p in model.parameters()), len(ys["train"])))

        best, best_state, bad = -1.0, None, 0
        for epoch in range(args.epochs):
            model.train()
            t0 = time.time()
            for x, y in loaders["train"]:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                loss = lossfn(model(x), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            vl, vy = predict(model, loaders["val"], device)
            va = float((vl.argmax(1) == vy).float().mean())
            if epoch % 5 == 0 or epoch == args.epochs - 1:
                print("    epoch %2d  val_acc %.4f  (%.0fs)"
                      % (epoch, va, time.time() - t0))
            if va > best:
                best, bad = va, 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    print("    early stop at epoch %d" % epoch)
                    break

        model.load_state_dict(best_state)
        vl, vy = predict(model, loaders["val"], device)
        temp = fit_temperature(vl, vy)

        out = {"temperature": temp, "val_acc": best}
        for split in ("calib", "test"):
            logits, y = predict(model, loaders[split], device)
            probs = torch.softmax(logits / temp, dim=1).numpy()
            out["%s_probs" % split] = probs
            out["%s_labels" % split] = y.numpy()
            out["%s_runs" % split] = runs[split]
        acc = float((out["test_probs"].argmax(1) == out["test_labels"]).mean())
        accs.append(acc)
        print("    val %.4f   test %.4f   temperature %.3f" % (best, acc, temp))

        np.savez_compressed(
            os.path.join(out_dir, "probs_fold%d.npz" % fold), **out)

    print("\nGRU test accuracy over %d folds: %.4f +- %.4f"
          % (len(accs), float(np.mean(accs)), float(np.std(accs))))
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2)
    print("Wrote %s" % out_dir)


if __name__ == "__main__":
    main()
