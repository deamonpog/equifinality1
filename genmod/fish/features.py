"""Per-(agent, event) features for the schooling-fish case study.

A run is a [T, 5, 2] position tensor on a shared but IRREGULAR time grid, so
every derivative divides by the actual inter-kick interval dt[t], never by a
constant. Two events are lost to the second derivative (velocity needs one
lag, acceleration needs two), and a burn-in drops the startup transient.

The feature block is split in two so that ablation 3 (basic vs relative
features) is a column slice rather than a re-run of preprocessing:

  BASIC    (11 cols) single-agent kinematics: where the agent is, how it is
           moving, how close it is to the wall, and how long since its last
           kick.
  RELATIVE  (9 cols) the relational signal the rules actually differ in.
           Every one is a SORTED or AGGREGATED function of the four
           neighbors, never indexed by agent id, so the block is permutation
           invariant by construction and not merely by augmentation.

Headings are stored as (sin, cos) rather than as an angle so the model never
sees the artificial discontinuity at +/- pi. Bearings to neighbors are
expressed in the FOCAL AGENT'S OWN FRAME (rotated by -theta_i), which makes
them invariant to global rotation of the arena, so rotation augmentation
cannot break them.
"""

import numpy as np

BASIC_FEATURES = [
    "x",            # position, normalized by arena radius
    "y",
    "vx",           # velocity, arena radii per second
    "vy",
    "speed",
    "sin_theta",    # heading
    "cos_theta",
    "ax",           # acceleration, arena radii per second squared
    "ay",
    "d_wall",       # 1 - r, so 1 at the centre and 0 at the wall
    "dt",           # seconds since the previous kick
]

RELATIVE_FEATURES = [
    "d_nn1",             # distance to the nearest neighbor
    "d_nn2",             # distance to the second-nearest neighbor
    "d_neigh_mean",      # mean distance to the four neighbors
    "d_neigh_std",       # spread of neighbor distances
    "sin_dtheta_mean",   # mean heading of the neighbors, in the focal frame
    "cos_dtheta_mean",
    "sin_bearing_cent",  # direction to the group centroid, in the focal frame
    "cos_bearing_cent",
    "local_polar",       # |mean unit velocity of the four neighbors|
]

# Why no "bearing to the NEAREST neighbor" feature.
#
# Raw positions are quantized to 1 mm, and under the cohesive rules the
# agents pack so tightly that d_nn1 reaches exactly 0. Two neighbors are
# therefore often EXACTLY equidistant, which makes the identity of "the
# nearest neighbor" genuinely ambiguous: an argmin over tied distances is
# decided by floating-point noise. Any feature indexed by that argmin jumps
# discontinuously when the tie flips, so it injects noise into training and
# breaks rotation equivariance (scripts/fish/verify_augmentation.py catches
# exactly this).
#
# The angular features are therefore AGGREGATES over all four neighbors, and
# the bearing is taken to the group centroid. Both are continuous under ties.
# The SORTED DISTANCES d_nn1 and d_nn2 are kept: a sorted value is continuous
# even when the index that produced it is not.

FEATURE_NAMES = BASIC_FEATURES + RELATIVE_FEATURES
N_BASIC = len(BASIC_FEATURES)
N_FEATURES = len(FEATURE_NAMES)

# The subset that is INVARIANT under a global rotation of the arena.
#
# x, y, vx, vy, ax, ay and the heading (sin, cos) all rotate with the frame.
# A model fed those must learn, from rotation augmentation alone, to ignore an
# arbitrary global angle -- capacity spent on a nuisance direction rather than
# on the rule. The remaining features are lengths, or angles already expressed
# in the focal agent's own frame, so they carry no absolute direction at all.
#
# This is what lets the trajectory Transformer meet the GRU baseline on equal
# terms: the GRU's group statistics are invariant BY CONSTRUCTION, so handing
# the Transformer raw coordinates and then comparing the two conflates "agent
# structure does not help" with "the Transformer had to learn invariances the
# GRU was given for free".
#
# Acceleration magnitude is not listed and does not need to be: the temporal
# attention sees speed at consecutive events and can form it directly.
INVARIANT_FEATURES = [
    "speed", "d_wall", "dt",
    "d_nn1", "d_nn2", "d_neigh_mean", "d_neigh_std",
    "sin_dtheta_mean", "cos_dtheta_mean",
    "sin_bearing_cent", "cos_bearing_cent",
    "local_polar",
]
INVARIANT_COLS = [FEATURE_NAMES.index(n) for n in INVARIANT_FEATURES]

# Events dropped from the start of every run. Step 01 found that all ten
# out-of-arena positions occur within the first 2 s, a placement transient,
# and the shared-seed coupling between rule files lives in exactly the same
# early region. Burning in past it removes both at a cost of ~3% of events.
BURN_IN_EVENTS = 50


def _unit(v, eps=1e-9):
    """Row-wise unit vectors, safe at zero speed."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def compute_features(time, pos, radius=0.25, burn_in=BURN_IN_EVENTS):
    """Turn one run into a [T', 5, F] feature tensor.

    Args:
        time: [T] kick times in seconds, strictly increasing.
        pos:  [T, 5, 2] positions in metres.
        radius: arena radius in metres, used to normalize lengths.
        burn_in: events to drop from the start, after derivatives.

    Returns:
        feats: [T', 5, F] float32, T' = T - 2 - burn_in
        keep_time: [T'] the kick times the rows correspond to
    """
    t = np.asarray(time, dtype=np.float64)
    p = np.asarray(pos, dtype=np.float64) / radius  # arena-radius units

    dt = np.diff(t)                       # [T-1]
    dt = np.maximum(dt, 1e-6)             # step 01 proved dt > 0; guard anyway

    # Velocity at event k uses the displacement from k-1 to k, so it is
    # defined for k = 1..T-1. Acceleration needs one more lag: k = 2..T-1.
    vel = np.diff(p, axis=0) / dt[:, None, None]          # [T-1, 5, 2]
    acc = np.diff(vel, axis=0) / dt[1:, None, None]       # [T-2, 5, 2]

    # Align everything onto the acceleration grid: events 2..T-1.
    p = p[2:]                 # [T-2, 5, 2]
    vel = vel[1:]             # [T-2, 5, 2]
    dt = dt[1:]               # [T-2]
    t = t[2:]                 # [T-2]

    # ---- basic, single-agent -----------------------------------------
    speed = np.linalg.norm(vel, axis=-1)                  # [T', 5]
    heading = _unit(vel)                                  # [T', 5, 2]
    cos_th, sin_th = heading[..., 0], heading[..., 1]
    r = np.linalg.norm(p, axis=-1)                        # [T', 5]
    d_wall = 1.0 - r

    basic = np.stack([
        p[..., 0], p[..., 1],
        vel[..., 0], vel[..., 1],
        speed,
        sin_th, cos_th,
        acc[..., 0], acc[..., 1],
        d_wall,
        np.broadcast_to(dt[:, None], speed.shape),
    ], axis=-1)                                           # [T', 5, 11]

    # ---- relative, neighbor-derived -----------------------------------
    # Pairwise displacements: diff[k, i, j] = p_j - p_i.
    diff = p[:, None, :, :] - p[:, :, None, :]            # [T', 5, 5, 2]
    dist = np.linalg.norm(diff, axis=-1)                  # [T', 5, 5]

    # Mask the self-pair so it never enters the neighbor statistics.
    n_agents = p.shape[1]
    eye = np.eye(n_agents, dtype=bool)
    dist_masked = np.where(eye[None], np.inf, dist)

    # Sorted neighbor distances. These are VALUES, not indices, so they are
    # continuous even when two neighbors are exactly equidistant.
    sorted_d = np.sort(dist_masked, axis=-1)[..., :n_agents - 1]
    d_nn1 = sorted_d[..., 0]
    d_nn2 = sorted_d[..., 1]
    d_mean = sorted_d.mean(axis=-1)
    d_std = sorted_d.std(axis=-1)

    # Mean neighbor heading, expressed in the focal agent's frame: rotate each
    # neighbor's unit heading by -theta_i and average over the four neighbors.
    # Averaging (rather than picking the nearest) is what makes this tie-free.
    total_dir = heading.sum(axis=1, keepdims=True)         # [T', 1, 2]
    neigh_dir_mean = (total_dir - heading) / (n_agents - 1)  # [T', 5, 2]
    ncos, nsin = neigh_dir_mean[..., 0], neigh_dir_mean[..., 1]
    cos_dth = cos_th * ncos + sin_th * nsin
    sin_dth = cos_th * nsin - sin_th * ncos

    # Bearing to the group centroid, also rotated into the focal frame. The
    # centroid is a smooth function of all five positions, so unlike "the
    # direction to my nearest neighbor" it has no discontinuity.
    centroid = p.mean(axis=1, keepdims=True)               # [T', 1, 2]
    to_cent = _unit(centroid - p)                          # [T', 5, 2]
    cos_b = cos_th * to_cent[..., 0] + sin_th * to_cent[..., 1]
    sin_b = cos_th * to_cent[..., 1] - sin_th * to_cent[..., 0]

    # Local polarization: how aligned the four neighbors are, seen from i.
    local_polar = np.linalg.norm(neigh_dir_mean, axis=-1)

    relative = np.stack([
        d_nn1, d_nn2, d_mean, d_std,
        sin_dth, cos_dth,
        sin_b, cos_b,
        local_polar,
    ], axis=-1)                                           # [T', 5, 9]

    feats = np.concatenate([basic, relative], axis=-1).astype(np.float32)

    if burn_in > 0:
        feats = feats[burn_in:]
        t = t[burn_in:]

    return feats, t
