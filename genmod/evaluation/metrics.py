"""Classification metrics for ECA rule identification."""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def eval_topk(
    model,
    loader: DataLoader,
    device: str,
    k_list: Tuple[int, ...] = (1, 3, 5),
) -> Dict[int, float]:
    """Compute top-k accuracy on a dataloader."""
    model.eval()
    total = 0
    correct = {k: 0 for k in k_list}

    for batch in loader:
        x, tpos, spos, y = batch[0], batch[1], batch[2], batch[3]
        x, tpos, spos, y = x.to(device), tpos.to(device), spos.to(device), y.to(device)
        probs = model.probs(x, tpos, spos)
        total += y.size(0)

        for k in k_list:
            topk = torch.topk(probs, k=min(k, probs.size(-1)), dim=-1).indices
            hit = (topk == y.unsqueeze(-1)).any(dim=-1).sum().item()
            correct[k] += hit

    return {k: correct[k] / max(total, 1) for k in k_list}


@torch.no_grad()
def compute_nll(model, loader: DataLoader, device: str) -> float:
    """Compute average negative log-likelihood on a dataloader."""
    model.eval()
    total_nll = 0.0
    total_samples = 0

    for batch in loader:
        x, tpos, spos, y = batch[0], batch[1], batch[2], batch[3]
        x, tpos, spos, y = x.to(device), tpos.to(device), spos.to(device), y.to(device)
        logits = model(x, tpos, spos)
        nll = F.cross_entropy(logits, y, reduction="sum")
        total_nll += nll.item()
        total_samples += y.size(0)

    return total_nll / max(total_samples, 1)


@torch.no_grad()
def per_class_accuracy(
    model,
    loader: DataLoader,
    device: str,
    num_classes: int,
) -> Dict[int, float]:
    """Compute per-class top-1 accuracy."""
    model.eval()
    correct = np.zeros(num_classes)
    total = np.zeros(num_classes)

    for batch in loader:
        x, tpos, spos, y = batch[0], batch[1], batch[2], batch[3]
        x, tpos, spos, y = x.to(device), tpos.to(device), spos.to(device), y.to(device)
        preds = model(x, tpos, spos).argmax(dim=-1)
        for cls in range(num_classes):
            mask = y == cls
            total[cls] += mask.sum().item()
            correct[cls] += (preds[mask] == cls).sum().item()

    return {i: correct[i] / max(total[i], 1) for i in range(num_classes)}


@torch.no_grad()
def collect_predictions(
    model,
    loader: DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect all predictions and labels from a dataloader.

    Returns: (all_probs, all_preds, all_labels)
        all_probs: (N, C) softmax probabilities
        all_preds: (N,) predicted class indices
        all_labels: (N,) true class indices
    """
    model.eval()
    all_probs = []
    all_labels = []

    for batch in loader:
        x, tpos, spos, y = batch[0], batch[1], batch[2], batch[3]
        x, tpos, spos, y = x.to(device), tpos.to(device), spos.to(device), y.to(device)
        probs = model.probs(x, tpos, spos)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = all_probs.argmax(axis=1)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_probs, all_preds, all_labels


def confusion_matrix(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Compute confusion matrix (true x predicted)."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(labels, preds):
        cm[true, pred] += 1
    return cm


def mean_reciprocal_rank(probs: np.ndarray, labels: np.ndarray) -> float:
    """Compute mean reciprocal rank."""
    ranks = np.argsort(-probs, axis=1)
    mrr = 0.0
    for i, label in enumerate(labels):
        rank = np.where(ranks[i] == label)[0][0] + 1
        mrr += 1.0 / rank
    return mrr / len(labels)
