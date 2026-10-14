"""Embedding extraction, visualization (t-SNE/UMAP), and linear probes."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
except ImportError:
    plt = None

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


@torch.no_grad()
def extract_all_embeddings(
    model,
    loader: DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract [CLS] embeddings for all samples in a dataloader.

    Returns: (embeddings, labels)
        embeddings: (N, d_model)
        labels: (N,) label indices
    """
    model.eval()
    all_embs = []
    all_labels = []

    for batch in loader:
        x, tpos, spos, y = batch[0], batch[1], batch[2], batch[3]
        x, tpos, spos = x.to(device), tpos.to(device), spos.to(device)
        embs = model.extract_embeddings(x, tpos, spos)
        all_embs.append(embs.cpu().numpy())
        all_labels.append(y.numpy())

    return np.concatenate(all_embs), np.concatenate(all_labels)


def compute_tsne(
    embeddings: np.ndarray,
    perplexity: int = 30,
    seed: int = 42,
) -> np.ndarray:
    """t-SNE projection to 2D."""
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed, init="pca")
    return tsne.fit_transform(embeddings)


def compute_umap(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """UMAP projection to 2D. Requires umap-learn package."""
    if not HAS_UMAP:
        raise ImportError("umap-learn not installed. Install with: pip install umap-learn")
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed)
    return reducer.fit_transform(embeddings)


def linear_probe(
    embeddings: np.ndarray,
    targets: np.ndarray,
    cv: int = 5,
    max_iter: int = 1000,
) -> Dict[str, float]:
    """Train a linear probe on frozen embeddings using cross-validation.

    Returns: dict with "accuracy_mean", "accuracy_std"
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(embeddings)

    clf = LogisticRegression(max_iter=max_iter, solver="lbfgs")
    scores = cross_val_score(clf, X, targets, cv=cv, scoring="accuracy")

    return {
        "accuracy_mean": scores.mean(),
        "accuracy_std": scores.std(),
    }


def plot_embedding_space(
    coords_2d: np.ndarray,
    labels: np.ndarray,
    color_values: Optional[np.ndarray] = None,
    color_label: str = "Rule",
    title: str = "Embedding Space",
    cmap: str = "tab20",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    alpha: float = 0.6,
    s: float = 10,
):
    """Plot 2D embedding space colored by given values."""
    if plt is None:
        raise ImportError("matplotlib required for plotting")

    fig, ax = plt.subplots(figsize=figsize)

    if color_values is None:
        color_values = labels

    unique_vals = np.unique(color_values)
    if len(unique_vals) <= 20:
        # Categorical coloring
        colormap = plt.get_cmap(cmap)
        for i, val in enumerate(unique_vals):
            mask = color_values == val
            ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                       c=[colormap(i / max(len(unique_vals) - 1, 1))],
                       label=f"{color_label}={val}", alpha=alpha, s=s)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=6, ncol=2)
    else:
        # Continuous coloring
        scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1],
                             c=color_values, cmap=cmap, alpha=alpha, s=s)
        plt.colorbar(scatter, ax=ax, label=color_label)

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax
