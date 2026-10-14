"""Batched augmentation and scaling, as GPU tensor ops.

The first implementation did this per window in numpy inside DataLoader
workers, and the GPU sat at about 10% utilisation. Two causes, both fixed
here:

  worker copies   Each of the 4 workers received its own copy of the 283 MB
                  feature tensor, respawned every epoch. Batches are now cut
                  in the main process with vectorized fancy indexing, which
                  is cheap, so no workers are needed at all.

  per-window math Rotating a [T, 5, 20] window in numpy, one window at a
                  time, is dominated by Python overhead. The same rotation
                  over a whole [B, T, 5, 20] batch on the GPU is a handful of
                  elementwise kernels.

The transformations are identical to the numpy ones in genmod/fish/augment.py,
which remain the reference implementation;
scripts/fish/verify_gpu_transform.py asserts the two agree.

Order is still: raw window -> augment -> scale. Augmentation must precede
scaling because a rotation mixes feature columns that the scaler treats
independently.
"""

import numpy as np
import torch

from genmod.fish.augment import REFLECTION_NEGATE, ROTATION_PAIRS


class GPUTransform:
    """Holds the fold's scaler on the device and applies it to raw batches."""

    def __init__(self, median, iqr, clip, device):
        self.median = torch.as_tensor(median, dtype=torch.float32, device=device)
        self.iqr = torch.as_tensor(iqr, dtype=torch.float32, device=device)
        self.clip = float(clip)
        self.device = device

    def scale(self, x):
        """[B, T, A, F] raw -> robustly scaled and clipped."""
        return torch.clamp((x - self.median) / self.iqr, -self.clip, self.clip)

    def augment(self, x, rotate=True, reflect=True, permute=True, generator=None):
        """Apply the enabled symmetries to a RAW batch, independently per sample."""
        B, T, A, F = x.shape
        dev = x.device

        if permute:
            # A different agent relabelling for each sample in the batch, held
            # fixed across the sample's events: agent identities are relabelled
            # once, not reshuffled at every kick.
            perm = torch.argsort(
                torch.rand(B, A, device=dev, generator=generator), dim=1)
            idx = perm[:, None, :, None].expand(B, T, A, F)
            x = torch.gather(x, 2, idx)

        if rotate:
            phi = torch.rand(B, device=dev, generator=generator) * (2 * torch.pi)
            c = torch.cos(phi)[:, None, None]
            s = torch.sin(phi)[:, None, None]
            x = x.clone()
            for ix, iy in ROTATION_PAIRS:
                u = x[..., ix].clone()
                v = x[..., iy].clone()
                x[..., ix] = c * u - s * v
                x[..., iy] = s * u + c * v

        if reflect:
            # Half the batch, on average. A reflection flips handedness, so the
            # sine of every focal-frame angle flips with the y components.
            flip = (torch.rand(B, device=dev, generator=generator) < 0.5)
            sign = torch.where(flip, -1.0, 1.0)[:, None, None]
            x = x.clone()
            for j in REFLECTION_NEGATE:
                x[..., j] = x[..., j] * sign

        return x


def cut_batch(feats, offsets, index, rows, T):
    """Gather a [B, T, A, F] raw batch without a per-window Python loop.

    Builds the event indices for every window at once and lets numpy do a
    single fancy-index gather:

        event[b, t] = offsets[run[b]] + start[b] + t
    """
    run = index[rows, 0]
    start = index[rows, 1]
    base = offsets[run] + start                       # [B]
    ev = base[:, None] + np.arange(T)[None, :]        # [B, T]
    return feats[ev]                                  # [B, T, A, F]
