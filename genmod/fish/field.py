"""Render fish trajectories onto a coarse spatial field (a lossy theta_obs).

The trajectory representation in genmod/fish/features.py keeps per-agent
identity: the model can follow fish i from kick to kick. This module produces
the deliberately COARSER observation used in the field ablation: every kick
event is splatted onto a G x G grid, so the five agents are summed into a
density field and their identities are gone. It is a controlled degradation of
the observation, not an attempt at a better representation -- the point of the
ablation is to measure how much the conformal set size inflates once identity
and sub-cell geometry are discarded.

Three channels per cell:

    density      splat mass, so a frame carries about 5 units of total mass
                 (one per fish), spread over the cells each fish touches.
    momentum_x   density-weighted velocity, sum of weight * vx over the fish
    momentum_y   deposited in the cell, so heading survives coarsening.

Rendering is done on the GPU from the raw coordinate and velocity columns
(feats[..., 0:2] and feats[..., 2:4], both already in arena-radius units), one
field per kick event. Nothing is precomputed to disk: a precomputed field
tensor would be tens of gigabytes, and rendering on the fly lets augmentation
happen in COORDINATE space before splatting, reusing the arena symmetries.

Frames are kick events, NOT resampled onto a regular clock, so the irregular
inter-kick interval dt is preserved and carried alongside the field.
"""

import numpy as np
import torch


def _to_grid_coords(pos, G):
    """Arena-radius coords (~[-1, 1] inside the disk) -> continuous [0, G-1]."""
    gx = (pos[..., 0] * 0.5 + 0.5) * (G - 1)
    gy = (pos[..., 1] * 0.5 + 0.5) * (G - 1)
    gx = gx.clamp(0.0, float(G - 1))
    gy = gy.clamp(0.0, float(G - 1))
    return gx, gy


def render_field(pos, vel, G=32, sigma=0.60, splat="bilinear"):
    """Render [B, T, A, 2] positions/velocities to a [B, T, G, G, 3] field.

    Channels are [density, momentum_x, momentum_y]. Mass is conserved per fish
    (the deposit weights of each fish sum to 1), so the total density of a
    frame is A minus any mass a fish outside the disk contributes at the
    clamped border. Row index is the y coordinate, column index is x.
    """
    B, T, A, _ = pos.shape
    device = pos.device
    gx, gy = _to_grid_coords(pos, G)              # [B, T, A]
    vx = vel[..., 0]
    vy = vel[..., 1]

    n_frames = B * T
    frame = torch.arange(n_frames, device=device).view(B, T, 1)  # [B, T, 1]
    dens = torch.zeros(n_frames * G * G, device=device)
    momx = torch.zeros(n_frames * G * G, device=device)
    momy = torch.zeros(n_frames * G * G, device=device)

    def deposit(xi, yi, w):
        """Add weight w (and weighted momentum) at integer cell (col=xi,row=yi)."""
        cell = yi * G + xi                        # [B, T, A]
        gidx = (frame * (G * G) + cell).reshape(-1)
        dens.index_add_(0, gidx, w.reshape(-1))
        momx.index_add_(0, gidx, (w * vx).reshape(-1))
        momy.index_add_(0, gidx, (w * vy).reshape(-1))

    if splat == "bilinear":
        x0 = torch.floor(gx)
        y0 = torch.floor(gy)
        wx = gx - x0
        wy = gy - y0
        x0l = x0.long().clamp(0, G - 1)
        x1l = (x0.long() + 1).clamp(0, G - 1)
        y0l = y0.long().clamp(0, G - 1)
        y1l = (y0.long() + 1).clamp(0, G - 1)
        deposit(x0l, y0l, (1.0 - wx) * (1.0 - wy))
        deposit(x1l, y0l, wx * (1.0 - wy))
        deposit(x0l, y1l, (1.0 - wx) * wy)
        deposit(x1l, y1l, wx * wy)
    elif splat == "gaussian":
        r = max(1, int(np.ceil(3.0 * sigma)))
        xc = torch.round(gx).long()
        yc = torch.round(gy).long()
        # First pass: per-point normalizer so each fish deposits unit mass.
        wsum = torch.zeros_like(gx)
        deposits = []
        for dxo in range(-r, r + 1):
            for dyo in range(-r, r + 1):
                xi = xc + dxo
                yi = yc + dyo
                inb = ((xi >= 0) & (xi < G) & (yi >= 0) & (yi < G)).float()
                dxf = (xc + dxo).float() - gx
                dyf = (yc + dyo).float() - gy
                w = torch.exp(-(dxf * dxf + dyf * dyf) / (2.0 * sigma * sigma))
                w = w * inb
                wsum = wsum + w
                deposits.append((xi.clamp(0, G - 1), yi.clamp(0, G - 1), w))
        wsum = wsum.clamp_min(1e-12)
        for xi, yi, w in deposits:
            deposit(xi, yi, w / wsum)
    else:
        raise ValueError("unknown splat %r" % splat)

    field = torch.stack([dens, momx, momy], dim=-1).reshape(B, T, G, G, 3)
    return field


def augment_positions(pos, vel, rotate=True, reflect=True, generator=None):
    """Apply arena symmetries to raw [B, T, A, 2] coords/vels, per sample.

    Rotation and reflection are the same symmetries genmod/fish/augment.py
    applies to the trajectory features; here they act directly on the 2-vectors
    before splatting. Agent permutation is intentionally omitted: the field is
    permutation invariant by construction (the splat sums over agents).
    """
    B = pos.shape[0]
    device = pos.device
    pos = pos.clone()
    vel = vel.clone()

    if rotate:
        phi = torch.rand(B, device=device, generator=generator) * (2.0 * torch.pi)
        c = torch.cos(phi).view(B, 1, 1)
        s = torch.sin(phi).view(B, 1, 1)
        px, py = pos[..., 0], pos[..., 1]
        pos = torch.stack([c * px - s * py, s * px + c * py], dim=-1)
        vx, vy = vel[..., 0], vel[..., 1]
        vel = torch.stack([c * vx - s * vy, s * vx + c * vy], dim=-1)

    if reflect:
        flip = torch.rand(B, device=device, generator=generator) < 0.5
        sign = torch.where(flip, -1.0, 1.0).view(B, 1, 1)
        pos = torch.stack([pos[..., 0], pos[..., 1] * sign], dim=-1)
        vel = torch.stack([vel[..., 0], vel[..., 1] * sign], dim=-1)

    return pos, vel


class FieldStandardizer:
    """Per-channel standardization fitted on TRAIN fields only.

    Continuous fields need a mean/std scaler rather than the robust median/IQR
    used for the trajectory features. Fit by streaming the fold's training
    windows through render_field once and accumulating moments; this mirrors
    the "fit the scaler on the training experiments only" rule in
    genmod/fish/windows.py, and needs no new data file.
    """

    def __init__(self, clip=6.0):
        self.clip = float(clip)
        self.mean = None
        self.std = None
        self._n = 0
        self._s = None
        self._s2 = None

    def update(self, field):
        flat = field.reshape(-1, field.shape[-1]).double()
        if self._s is None:
            self._s = torch.zeros(flat.shape[-1], device=flat.device,
                                  dtype=torch.double)
            self._s2 = torch.zeros_like(self._s)
        self._n += flat.shape[0]
        self._s += flat.sum(0)
        self._s2 += (flat * flat).sum(0)
        return self

    def finalize(self):
        mean = self._s / self._n
        var = (self._s2 / self._n - mean * mean).clamp_min(1e-12)
        self.mean = mean.float()
        self.std = torch.sqrt(var).float()
        return self

    def apply(self, field):
        z = (field - self.mean) / self.std
        return torch.clamp(z, -self.clip, self.clip)
