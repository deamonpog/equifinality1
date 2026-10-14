"""Symmetry augmentations for [T, 5, F] trajectory windows.

The generating rules are invariant under three transformations, so a model
that is not invariant to them is spending capacity learning nuisance
structure:

  agent permutation   the five agents are behaviorally interchangeable, so
                      the rule cannot depend on how they are numbered
  global rotation     the arena is a circle, so no direction is special
  reflection          the rules contain no left/right handedness

Time reversal is NOT a symmetry and is never applied: the dynamics are
causal, and a reversed trajectory does not obey the same rule.

These act on the RAW feature vector, and must be applied BEFORE the robust
scaler, because a rotation mixes feature columns that the scaler treats
independently. Rotating after scaling would combine differently-scaled
quantities and produce something that is not a rotated trajectory at all.

The transformations do not act uniformly on the feature vector, which is the
easy thing to get wrong:

  rotation   acts on the 2-vectors (x,y), (vx,vy), (ax,ay), (cos_t, sin_t).
             Everything else -- distances, local_polar, and the focal-frame
             bearings -- is already rotation invariant by construction and
             must be left alone.

  reflection negates the y components, and ALSO negates the sine of every
             focal-frame angle. Reflection flips handedness, so "the nearest
             neighbor is 30 degrees to my left" becomes "30 degrees to my
             right"; the cosine is unchanged but the sine flips. Missing this
             would teach the model a chirality that is not in the data.
"""

import numpy as np

from genmod.fish.features import FEATURE_NAMES

_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}

# Feature pairs that transform as 2-vectors under rotation, as
# (x_like, y_like). The heading is stored as (sin, cos), so its vector form
# is (cos_theta, sin_theta).
ROTATION_PAIRS = [
    (_IDX["x"], _IDX["y"]),
    (_IDX["vx"], _IDX["vy"]),
    (_IDX["ax"], _IDX["ay"]),
    (_IDX["cos_theta"], _IDX["sin_theta"]),
]

# Negated by a reflection about the x axis: the y components, plus the sine
# of every angle expressed in the focal agent's frame.
REFLECTION_NEGATE = [
    _IDX["y"],
    _IDX["vy"],
    _IDX["ay"],
    _IDX["sin_theta"],
    _IDX["sin_dtheta_mean"],    # handedness of the neighbors' relative heading
    _IDX["sin_bearing_cent"],   # handedness of the direction to the centroid
]

# Features that must be untouched by rotation and reflection: lengths and
# quantities already expressed in a rotation-invariant frame. Listed so
# scripts/fish/verify_augmentation.py can assert they really are invariant.
INVARIANT = ["speed", "d_wall", "dt", "d_nn1", "d_nn2", "d_neigh_mean",
             "d_neigh_std", "cos_dtheta_mean", "cos_bearing_cent",
             "local_polar"]


def rotate(win, phi):
    """Rotate a [T, 5, F] window by angle phi about the arena centre."""
    out = win.copy()
    c, s = np.cos(phi), np.sin(phi)
    for ix, iy in ROTATION_PAIRS:
        u, v = win[..., ix], win[..., iy]
        out[..., ix] = c * u - s * v
        out[..., iy] = s * u + c * v
    return out


def reflect(win):
    """Reflect a [T, 5, F] window across the x axis."""
    out = win.copy()
    out[..., REFLECTION_NEGATE] *= -1.0
    return out


def permute_agents(win, rng):
    """Randomly relabel the five agents."""
    return win[:, rng.permutation(win.shape[1]), :]


def augment(win, rng, do_rotate=True, do_reflect=True, do_permute=True):
    """Apply the enabled symmetries to one raw window."""
    if do_permute:
        win = permute_agents(win, rng)
    if do_rotate:
        win = rotate(win, rng.uniform(0.0, 2.0 * np.pi))
    if do_reflect and rng.random() < 0.5:
        win = reflect(win)
    return win
