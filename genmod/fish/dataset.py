"""Torch datasets over the window index.

Windows are cut lazily from the shared feature tensor rather than
materialized, so all six window lengths and five folds cost one 283 MB array
in memory instead of tens of gigabytes of overlapping copies.

Order of operations, which matters:

    slice raw window  ->  augment  ->  scale

Augmentation must precede scaling. A rotation mixes feature columns (x with
y, vx with vy), while the robust scaler treats every column independently
with its own median and IQR. Scaling first and rotating second would combine
differently-scaled quantities and produce something that is not a rotated
trajectory at all.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from genmod.fish.augment import augment
from genmod.fish.windows import apply_scaler


class FishWindowDataset(Dataset):
    """Agent-time windows: [T, 5, F] features and the generating rule."""

    def __init__(self, feats, offsets, index, labels, median, iqr, T,
                 feature_cols=None, augment_cfg=None, seed=0):
        """
        Args:
            feats:    [E, 5, F] raw feature tensor shared by all splits.
            offsets:  [n_runs + 1] event offsets into feats.
            index:    [N, 2] (run_index, start_event) for this split.
            labels:   [n_runs] rule id of each run.
            median, iqr: the fold's scaler, fitted on TRAIN runs only.
            T:        window length in events.
            feature_cols: optional column subset (ablation 3: basic only).
            augment_cfg:  dict(rotate=, reflect=, permute=) or None for eval.
        """
        self.feats = feats
        self.offsets = offsets
        self.index = index
        self.labels = labels
        self.T = T
        self.cols = feature_cols
        self.augment_cfg = augment_cfg
        self.median = median
        self.iqr = iqr
        self.seed = seed

    def __len__(self):
        return self.index.shape[0]

    def __getitem__(self, i):
        run_idx, start = self.index[i]
        o = self.offsets[run_idx]
        win = self.feats[o + start:o + start + self.T]        # [T, 5, F] raw

        if self.augment_cfg is not None:
            # A fresh stream per item and epoch would be ideal; seeding on the
            # item index alone would replay the same rotation every epoch and
            # silently turn augmentation into a fixed relabelling of the data.
            rng = np.random.default_rng(
                (self.seed, int(run_idx), int(start), torch.initial_seed() & 0xFFFF))
            win = augment(win, rng, **self.augment_cfg)

        win = apply_scaler(win, self.median, self.iqr)

        if self.cols is not None:
            win = win[..., self.cols]

        return (torch.from_numpy(np.ascontiguousarray(win, dtype=np.float32)),
                torch.tensor(int(self.labels[run_idx]), dtype=torch.long))

    def run_indices(self):
        """The run each window came from, for run-level bootstrap CIs."""
        return self.index[:, 0]


class GroupSequenceDataset(Dataset):
    """Per-event GROUP statistics: [T, G] and the generating rule.

    The GRU baseline's view of the world. Agents are aggregated away at every
    event, so individual identity is gone but the time course is preserved.
    """

    def __init__(self, X, y, runs):
        self.X = X          # [N, T, G] float32, already scaled
        self.y = y          # [N]
        self.runs = runs    # [N] run index, for bootstrap CIs

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return (torch.from_numpy(self.X[i]),
                torch.tensor(int(self.y[i]), dtype=torch.long))

    def run_indices(self):
        return self.runs
