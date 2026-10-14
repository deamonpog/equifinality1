"""Data splitting logic for train/val/test.

Provides run-level 3-way splits within each class.
"""

from typing import Dict, Tuple

import torch


def split_runs(
    runs: list,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[list, list, list]:
    """Three-way split of runs for a single class.

    Returns: (train_runs, val_runs, test_runs)
    """
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(runs), generator=g).tolist()
    n = len(runs)
    n_train = max(1, int(train_frac * n))
    n_val = max(1, int(val_frac * n))
    # Ensure test has at least 1 run if possible
    if n_train + n_val >= n and n > 2:
        n_val = max(1, n - n_train - 1)

    train = [runs[i] for i in perm[:n_train]]
    val = [runs[i] for i in perm[n_train : n_train + n_val]]
    test = [runs[i] for i in perm[n_train + n_val :]]

    # Ensure val and test are non-empty
    if len(val) == 0 and len(train) > 1:
        val = [train.pop()]
    if len(test) == 0 and len(train) > 1:
        test = [train.pop()]

    return train, val, test


def prepare_splits(
    all_runs_by_label: Dict[int, list],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[Dict[int, list], Dict[int, list], Dict[int, list]]:
    """Split runs within each class into train/val/test.

    Returns: (train_by_label, val_by_label, test_by_label)
    """
    train_by_label: Dict[int, list] = {}
    val_by_label: Dict[int, list] = {}
    test_by_label: Dict[int, list] = {}

    for label, runs in all_runs_by_label.items():
        tr, va, te = split_runs(runs, train_frac, val_frac, seed=seed + label)
        train_by_label[label] = tr
        val_by_label[label] = va
        test_by_label[label] = te

    return train_by_label, val_by_label, test_by_label
