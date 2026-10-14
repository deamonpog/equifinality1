"""Training script for rule-tree classification.

System: rule_trees (Schelling segregation with auto-generated rule trees).
Uses a Transformer encoder over patch tokens with decomposed time/space
positional embeddings, plus RAPS conformal prediction for uncertainty
quantification.

The conformal block at the end of training is a quick check only: it
calibrates on part of the VALIDATION split, which was also used for early
stopping and temperature scaling, so its scores are not exchangeable with the
test scores. The conformal sets reported in the paper (Table 1) come from
scripts/recalibrate_conformal.py, which calibrates on half of the held-out
TEST split and evaluates on the other half.

Usage:
    python scripts/train.py --config configs/ruletree_base.yaml
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genmod.models.factory import build_model, get_tokenization_params
from genmod.evaluation.metrics import eval_topk, compute_nll, per_class_accuracy, collect_predictions
from genmod.evaluation.calibration import (
    expected_calibration_error, calibrate_temperature,
    reliability_diagram, brier_score,
)
from genmod.evaluation.embeddings import extract_all_embeddings, compute_tsne, linear_probe, plot_embedding_space
from genmod.utils.config import load_config, ExperimentConfig


# ============================================================
# System-specific data loading
# ============================================================

def _load_rule_trees(cfg):
    """Load rule-tree Schelling data and build datasets."""
    from genmod.data.schelling_dataset import SchellingDataset
    from genmod.data.ruletree_dataset import load_ruletree_data, get_ruletree_tokenization_params
    from genmod.data.rule_trees import load_tree_library
    from genmod.data.ruletree_metadata import get_tree_group
    from genmod.data.splits import split_runs

    data_root = Path(cfg.data.data_dir) / "rule_trees"
    library = load_tree_library(data_root / "tree_library.json")
    n_classes = len(library)
    print(f"  Loaded {n_classes} rule trees from {data_root / 'tree_library.json'}")

    all_runs = load_ruletree_data(data_root, n_classes)

    train_by_label, val_by_label, test_by_label = {}, {}, {}
    for label, runs in all_runs.items():
        tr, va, te = split_runs(runs, cfg.data.train_frac, cfg.data.val_frac,
                                seed=cfg.training.seed + label)
        train_by_label[label] = tr
        val_by_label[label] = va
        test_by_label[label] = te

    tok = get_ruletree_tokenization_params(
        grid_size=cfg.data.grid_size,
        patch_size=cfg.data.schelling_patch_size,
        num_snapshots=cfg.data.num_snapshots,
    )

    ds_kwargs = dict(patch_size=cfg.data.schelling_patch_size,
                     num_snapshots=cfg.data.num_snapshots,
                     grid_size=cfg.data.grid_size)
    train_ds = SchellingDataset(train_by_label, **ds_kwargs)
    val_ds = SchellingDataset(val_by_label, **ds_kwargs)
    test_ds = SchellingDataset(test_by_label, **ds_kwargs)

    label_to_tree_str = {label: tree_str for label, tree, tree_str in library}
    tree_by_label = {label: tree for label, tree, tree_str in library}

    def get_group(label):
        return get_tree_group(tree_by_label[label])

    return train_ds, val_ds, test_ds, n_classes, label_to_tree_str, tok, get_group, []


SYSTEM_LOADERS = {
    "rule_trees": _load_rule_trees,
}


# ============================================================
# Training
# ============================================================

def train(cfg: ExperimentConfig):
    torch.manual_seed(cfg.training.seed)
    device = cfg.training.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, using CPU")

    ckpt_dir = Path(cfg.training.checkpoint_dir) / cfg.name
    log_dir = Path(cfg.training.log_dir) / cfg.name
    fig_dir = Path("results/figures") / cfg.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    system = cfg.data.system
    print(f"System: {system}")
    loader_fn = SYSTEM_LOADERS.get(system)
    if loader_fn is None:
        raise ValueError(f"Unknown system: {system}. Choose from {list(SYSTEM_LOADERS)}")

    train_ds, val_ds, test_ds, num_classes, label_mapping, tok_params, get_group_fn, held_out = loader_fn(cfg)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}, Classes: {num_classes}")

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False)

    model = build_model(
        num_classes=num_classes,
        vocab_size=tok_params["vocab_size"],
        seq_len=tok_params["seq_len"],
        time_size=tok_params["time_size"],
        space_size=tok_params["space_size"],
        model_size=cfg.model.model_size,
        dropout=cfg.model.dropout,
        positional_encoding=cfg.model.positional_encoding,
        grid_rows=tok_params.get("grid_rows"),
        grid_cols=tok_params.get("grid_cols"),
    ).to(device)

    n_params = model.count_parameters()
    print(f"Model: {cfg.model.model_size} | Params: {n_params:,} | SeqLen: {tok_params['seq_len']}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    total_steps = cfg.training.epochs * len(train_loader)
    warmup_steps = cfg.training.warmup_epochs * len(train_loader)

    if cfg.training.lr_scheduler == "cosine":
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    log_file = log_dir / "training_log.csv"
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_top1", "val_top3", "val_top5", "val_nll", "lr", "time_s"])

    best_val_acc = 0.0

    for epoch in range(1, cfg.training.epochs + 1):
        t0 = time.time()
        model.train()
        epoch_loss, n_batches = 0.0, 0

        for x, tpos, spos, y in train_loader:
            x, tpos, spos, y = x.to(device), tpos.to(device), spos.to(device), y.to(device)
            logits = model(x, tpos, spos)
            loss = F.cross_entropy(logits, y, label_smoothing=cfg.training.label_smoothing)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            optimizer.step()
            if scheduler:
                scheduler.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        lr_now = optimizer.param_groups[0]["lr"]
        val_topk = eval_topk(model, val_loader, device)
        val_nll = compute_nll(model, val_loader, device)
        elapsed = time.time() - t0

        print(f"epoch {epoch:02d} | loss={avg_loss:.4f} | top1={val_topk[1]*100:.1f}% | "
              f"top3={val_topk[3]*100:.1f}% | top5={val_topk[5]*100:.1f}% | "
              f"nll={val_nll:.3f} | lr={lr_now:.2e} | {elapsed:.1f}s")

        with open(log_file, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{avg_loss:.4f}", f"{val_topk[1]:.4f}",
                                    f"{val_topk[3]:.4f}", f"{val_topk[5]:.4f}",
                                    f"{val_nll:.4f}", f"{lr_now:.6f}", f"{elapsed:.1f}"])

        if val_topk[1] > best_val_acc:
            best_val_acc = val_topk[1]
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                         "val_top1": val_topk[1], "label_mapping": label_mapping},
                        ckpt_dir / "best.pt")
        if epoch % cfg.training.save_every == 0:
            torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:03d}.pt")

    best = torch.load(ckpt_dir / "best.pt", weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    print(f"\nBest val top-1: {best_val_acc*100:.1f}% (epoch {best['epoch']})")

    # ==================
    # Post-training evaluation
    # ==================
    print("\n=== Test Set Evaluation ===")
    test_topk = eval_topk(model, test_loader, device)
    test_nll = compute_nll(model, test_loader, device)
    print(f"Test top-1={test_topk[1]*100:.1f}% | top-3={test_topk[3]*100:.1f}% | "
          f"top-5={test_topk[5]*100:.1f}% | nll={test_nll:.3f}")

    print("\n=== Calibration Analysis ===")
    all_probs, all_preds, all_labels = collect_predictions(model, test_loader, device)
    ece, _, _, _ = expected_calibration_error(all_probs, all_labels)
    bs = brier_score(all_probs, all_labels)
    print(f"ECE (pre-temp): {ece:.4f} | Brier: {bs:.4f}")

    learned_temp = calibrate_temperature(model, val_loader, device)
    print(f"Learned temperature: {learned_temp:.3f}")

    all_logits_list = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            x, tpos, spos = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            all_logits_list.append(model(x, tpos, spos).cpu())
    all_logits_tensor = torch.cat(all_logits_list)
    scaled_probs = torch.softmax(all_logits_tensor / learned_temp, dim=-1).numpy()
    ece_post, _, _, _ = expected_calibration_error(scaled_probs, all_labels)
    print(f"ECE (post-temp): {ece_post:.4f}")

    try:
        reliability_diagram(all_probs, all_labels, save_path=str(fig_dir / "reliability_pre_temp.png"),
                            title=f"{system} - Before Temp Scaling")
        reliability_diagram(scaled_probs, all_labels, save_path=str(fig_dir / "reliability_post_temp.png"),
                            title=f"{system} - After Temp Scaling")
    except Exception:
        pass

    print(f"\n=== Per-Group Accuracy ({system}) ===")
    class_accs = per_class_accuracy(model, test_loader, device, num_classes)
    group_correct, group_total = {}, {}
    for label_idx, acc in class_accs.items():
        g = get_group_fn(label_idx)
        group_correct[g] = group_correct.get(g, 0) + acc
        group_total[g] = group_total.get(g, 0) + 1
    for g in sorted(group_correct):
        if group_total[g] > 0:
            print(f"  Group {g}: {group_correct[g]/group_total[g]*100:.1f}% ({group_total[g]} classes)")

    conformal_results = {}
    if cfg.conformal.enabled:
        print("\n=== Conformal Prediction (quick check; see scripts/recalibrate_conformal.py) ===")
        from genmod.evaluation.conformal import run_conformal_at_multiple_alphas

        val_probs, _, val_labels = collect_predictions(model, val_loader, device)
        n_cal = int(len(val_labels) * cfg.conformal.cal_fraction)
        cal_probs, cal_labels = val_probs[:n_cal], val_labels[:n_cal]

        test_group_labels = np.array([get_group_fn(l) for l in all_labels])

        conformal_results = run_conformal_at_multiple_alphas(
            cal_probs, cal_labels, all_probs, all_labels, test_group_labels,
            alpha_levels=cfg.conformal.alpha_levels,
            method=cfg.conformal.method,
            k_reg=cfg.conformal.k_reg,
            lambda_reg=cfg.conformal.lambda_reg,
        )

        for alpha, res in sorted(conformal_results.items()):
            print(f"  alpha={alpha:.2f}: coverage={res['coverage']:.3f}, "
                  f"avg_set_size={res['avg_set_size']:.1f}, "
                  f"singleton={res['singleton_fraction']:.2f}")

    print("\n=== Embedding Analysis ===")
    embeddings, emb_labels = extract_all_embeddings(model, test_loader, device)
    print(f"Extracted {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")

    try:
        coords = compute_tsne(embeddings, perplexity=min(30, max(5, len(embeddings) // 5)))
        group_labels = np.array([get_group_fn(l) for l in emb_labels])
        plot_embedding_space(coords, emb_labels, color_values=group_labels,
                             color_label="Group", title=f"{system} t-SNE by Group",
                             cmap="Set1", save_path=str(fig_dir / "tsne_group.png"))
    except Exception as e:
        print(f"Skipping t-SNE: {e}")

    try:
        group_targets = np.array([get_group_fn(l) for l in emb_labels])
        if len(np.unique(group_targets)) > 1:
            probe = linear_probe(embeddings, group_targets)
            print(f"Group probe: {probe['accuracy_mean']*100:.1f}% (+/- {probe['accuracy_std']*100:.1f}%)")
        else:
            probe = {"accuracy_mean": 1.0, "accuracy_std": 0.0}
    except Exception:
        probe = {"accuracy_mean": 0.0, "accuracy_std": 0.0}

    results = {
        "system": system,
        "config": cfg.name,
        "model_size": cfg.model.model_size,
        "num_params": n_params,
        "num_classes": num_classes,
        "train_samples": len(train_ds),
        "test_top1": test_topk[1],
        "test_top3": test_topk[3],
        "test_top5": test_topk[5],
        "test_nll": test_nll,
        "ece_pre_temp": ece,
        "ece_post_temp": ece_post,
        "brier_score": float(bs),
        "learned_temperature": learned_temp,
        "group_probe_accuracy": probe["accuracy_mean"],
    }
    if conformal_results:
        results["conformal"] = {str(k): v for k, v in conformal_results.items()}

    with open(log_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {log_dir / 'results.json'}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train rule-tree classifier")
    parser.add_argument("--config", type=str, default="configs/ruletree_base.yaml")
    parser.add_argument("--model_size", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--positional_encoding", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--conformal", action="store_true", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.model_size:
        cfg.model.model_size = args.model_size
    if args.epochs:
        cfg.training.epochs = args.epochs
    if args.batch_size:
        cfg.training.batch_size = args.batch_size
    if args.positional_encoding:
        cfg.model.positional_encoding = args.positional_encoding
    if args.seed is not None:
        cfg.training.seed = args.seed
    if args.name:
        cfg.name = args.name
    if args.device:
        cfg.training.device = args.device
    if args.conformal:
        cfg.conformal.enabled = True

    train(cfg)


if __name__ == "__main__":
    main()
