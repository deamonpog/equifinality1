"""Event-indexed windowing and train-only feature scaling.

Windows are stored as INDICES, not as materialized tensors. A window is a
pair (run_index, start_event); the actual [T, 5, F] slice is cut from the
feature tensor in data/fish/runs.npz at load time. Four window lengths times
five folds times four partitions would otherwise cost tens of gigabytes of
near-duplicate float32, since overlapping windows share most of their events.

Two choices here carry statistical weight.

Stride. Training uses a dense stride to manufacture data volume. Calibration
and test use stride = T, so their windows are NON-OVERLAPPING. Overlapping
windows are not exchangeable: consecutive windows share most of their events
and their conformity scores are strongly dependent. Split conformal assumes
exchangeability between the calibration and test scores, and feeding it
overlapping windows produces a quantile that looks fine and a coverage
guarantee that is not actually valid. Dense stride in train is harmless
because train makes no such assumption.

Scaling. The scaler is fitted on the TRAINING experiments of each fold only,
never on all the data. Fitting on everything would leak the test-set
distribution into the model in a way that no training curve would reveal.
We use median and inter-quartile range rather than mean and standard
deviation, then clip: acceleration is a finite difference divided by an
inter-kick interval that can be as small as 2 ms, so it carries tails around
570 sigma that are artifacts of the irregular grid, not behavior. A
mean/std scaler would let those tails dictate the input scale.
"""

import numpy as np

CLIP = 8.0  # after robust scaling, in IQR units


def make_window_index(offsets, exp_ids, split_exps, T, stride):
    """Enumerate (run_index, start_event) for every window in a partition.

    Args:
        offsets:    [n_runs + 1] event offsets into the concatenated tensor.
        exp_ids:    [n_runs] experiment id of each run.
        split_exps: experiment ids belonging to this partition.
        T:          window length, in kick events.
        stride:     step between window starts, in kick events.

    Returns:
        [N, 2] int64 array of (run_index, start_event).
    """
    wanted = set(int(e) for e in split_exps)
    out = []
    for run_idx in range(len(offsets) - 1):
        if int(exp_ids[run_idx]) not in wanted:
            continue
        n_events = int(offsets[run_idx + 1] - offsets[run_idx])
        if n_events < T:
            continue
        for start in range(0, n_events - T + 1, stride):
            out.append((run_idx, start))
    return np.asarray(out, dtype=np.int64).reshape(-1, 2)


def fit_scaler(feats, offsets, train_run_idx):
    """Median and IQR of each feature, over the training runs only.

    Statistics are taken over every (event, agent) cell of the training runs,
    pooling the five agents, because the feature projection is shared across
    agents and must therefore see a single common scale.

    Returns:
        median: [F] float32
        iqr:    [F] float32, floored away from zero
    """
    chunks = [feats[offsets[i]:offsets[i + 1]] for i in train_run_idx]
    pooled = np.concatenate(chunks, axis=0).reshape(-1, feats.shape[-1])
    q25, median, q75 = np.percentile(pooled, [25, 50, 75], axis=0)
    iqr = np.maximum(q75 - q25, 1e-6)
    return median.astype(np.float32), iqr.astype(np.float32)


def apply_scaler(x, median, iqr, clip=CLIP):
    """Robustly scale and clip a feature array. Shape-agnostic in all but the
    last axis, which must be the feature axis."""
    z = (x - median) / iqr
    return np.clip(z, -clip, clip)
