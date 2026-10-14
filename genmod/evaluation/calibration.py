"""Calibration analysis: ECE, reliability diagrams, temperature scaling.

Reference: Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017).
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    strategy: str = "uniform",
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Expected Calibration Error.

    Args:
        probs: (N, C) predicted probabilities.
        labels: (N,) true labels.
        n_bins: Number of bins.
        strategy: "uniform" (equal-width) or "quantile" (equal-mass).

    Returns: (ece, bin_accuracies, bin_confidences, bin_counts)
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    if strategy == "uniform":
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
    elif strategy == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_boundaries = np.percentile(confidences, quantiles * 100)
        bin_boundaries[0] = 0.0
        bin_boundaries[-1] = 1.0
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=np.int64)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        count = mask.sum()
        bin_counts[i] = count
        if count > 0:
            bin_accs[i] = accuracies[mask].mean()
            bin_confs[i] = confidences[mask].mean()

    ece = np.sum(bin_counts * np.abs(bin_accs - bin_confs)) / max(labels.shape[0], 1)
    return ece, bin_accs, bin_confs, bin_counts


def maximum_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Compute Maximum Calibration Error (worst-bin miscalibration)."""
    _, bin_accs, bin_confs, bin_counts = expected_calibration_error(probs, labels, n_bins)
    nonzero = bin_counts > 0
    if not nonzero.any():
        return 0.0
    return np.max(np.abs(bin_accs[nonzero] - bin_confs[nonzero]))


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Compute multi-class Brier score."""
    n_classes = probs.shape[1]
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return np.mean(np.sum((probs - one_hot) ** 2, axis=1))


def reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    ax=None,
    title: str = "Reliability Diagram",
    save_path: Optional[str] = None,
):
    """Plot reliability diagram (calibration curve)."""
    if plt is None:
        raise ImportError("matplotlib required for plotting")

    ece, bin_accs, bin_confs, bin_counts = expected_calibration_error(probs, labels, n_bins)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")

    # Bar chart
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_widths = np.diff(bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    nonzero = bin_counts > 0
    ax.bar(bin_centers[nonzero], bin_accs[nonzero], width=bin_widths[nonzero],
           alpha=0.6, edgecolor="black", label="Observed")

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title} (ECE={ece:.4f})")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax


def calibrate_temperature(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    lr: float = 0.01,
    max_iter: int = 100,
) -> float:
    """Learn optimal temperature on validation set.

    Returns the learned temperature value.
    """
    # Collect all logits and labels
    all_logits = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            x, tpos, spos, y = batch[0], batch[1], batch[2], batch[3]
            x, tpos, spos, y = x.to(device), tpos.to(device), spos.to(device), y.to(device)
            logits = model(x, tpos, spos)
            all_logits.append(logits)
            all_labels.append(y)

    all_logits = torch.cat(all_logits).detach()
    all_labels = torch.cat(all_labels).detach()

    # Optimize log(temperature) to ensure temperature > 0
    log_temp = nn.Parameter(torch.zeros(1, device=device))
    optimizer = torch.optim.LBFGS([log_temp], lr=lr, max_iter=max_iter)

    def eval_fn():
        optimizer.zero_grad()
        temp = torch.exp(log_temp)
        scaled = all_logits / temp
        loss = F.cross_entropy(scaled, all_labels)
        loss.backward()
        return loss

    optimizer.step(eval_fn)

    learned_temp = torch.exp(log_temp).item()
    return learned_temp


def prediction_entropy(probs: np.ndarray) -> np.ndarray:
    """Per-sample entropy of predicted distribution.

    High entropy = model is uncertain about which rule generated the pattern.
    """
    # Clip for numerical stability
    p = np.clip(probs, 1e-10, 1.0)
    return -np.sum(p * np.log(p), axis=1)
