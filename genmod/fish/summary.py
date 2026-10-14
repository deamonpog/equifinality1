"""Window-level group statistics for the aggregate baselines.

These features deliberately DESTROY individual-agent identity: every quantity
is an average, an extremum, or a variance taken over the five agents. That is
the point. The trajectory Transformer keeps each agent as its own token and
can attend from one agent to another; this representation cannot. Comparing
the two isolates how much of the rule signal lives in the relational
structure rather than in the group's bulk statistics (RQ5, ablation 2).

Two families of statistic are computed:

  per-frame then aggregated
      A quantity is computed at each event (e.g. polarization of the group at
      time t), giving a time series of length T, which is then summarized by
      its mean and standard deviation over the window. The standard deviation
      matters: two rules can produce the same mean cohesion while differing
      in how much it fluctuates.

  pooled
      A quantity defined per (agent, event) cell is pooled over the whole
      window (e.g. mean speed).
"""

import numpy as np

SUMMARY_NAMES = [
    "speed_mean", "speed_std",
    "accel_mag_mean",
    "dt_mean", "dt_std",
    "d_nn1_mean", "d_nn1_std",
    "d_pair_mean", "d_pair_min",
    "group_radius_mean", "group_radius_std",
    "polarization_mean", "polarization_std",
    "ang_momentum_mean", "ang_momentum_abs_mean",
    "d_wall_mean", "d_wall_std",
    "heading_change_mean", "heading_change_std",
    "nn_persistence",
]

# The subset that involves NO neighbor relation whatsoever: how fast an agent
# swims, how hard it turns, how often it kicks, how close it sits to the wall.
# A rule can only be inferred from these through its indirect effect on solo
# kinematics. This is the deliberately weak rung of the ladder:
#
#   naive solo stats  ->  relational group stats  ->  agent-level Transformer
#
# It isolates how much of the signal comes from the relational features alone,
# before any agent-level structure is added.
NAIVE_NAMES = [
    "speed_mean", "speed_std",
    "accel_mag_mean",
    "dt_mean", "dt_std",
    "d_wall_mean", "d_wall_std",
    "heading_change_mean", "heading_change_std",
]



# Group state at a SINGLE event: the agents are aggregated away, but the time
# axis survives. This is the GRU baseline's view of the world, and it is the
# control that separates two explanations of any Transformer win. The GRU is
# also a sequence model and also sees the full time course; what it does not
# see is which agent is which. If the Transformer wins, the gain comes from
# preserving agent-level structure, not merely from modelling the sequence.
PER_EVENT_NAMES = [
    "speed_mean", "speed_std",
    "d_nn1_mean", "d_nn1_std",
    "d_pair_mean", "d_pair_min", "d_pair_max",
    "group_radius",
    "polarization",
    "ang_momentum",
    "d_wall_mean", "d_wall_min",
    "dt",
]


def per_event_group_features(win, feature_names):
    """Collapse a [T, 5, F] window into a [T, G] group-state sequence."""
    col = {n: i for i, n in enumerate(feature_names)}

    x = win[..., col["x"]]
    y = win[..., col["y"]]
    speed = win[..., col["speed"]]
    d_wall = win[..., col["d_wall"]]
    d_nn1 = win[..., col["d_nn1"]]
    dt = win[..., col["dt"]][:, 0]
    heading = np.stack([win[..., col["cos_theta"]], win[..., col["sin_theta"]]],
                       axis=-1)                              # [T, 5, 2]

    pos = np.stack([x, y], axis=-1)                          # [T, 5, 2]
    diff = pos[:, None, :, :] - pos[:, :, None, :]
    dist = np.linalg.norm(diff, axis=-1)
    iu = np.triu_indices(pos.shape[1], k=1)
    pair = dist[:, iu[0], iu[1]]                             # [T, 10]

    centroid = pos.mean(axis=1, keepdims=True)
    rel = pos - centroid
    group_radius = np.linalg.norm(rel, axis=-1).mean(axis=1)
    polarization = np.linalg.norm(heading.mean(axis=1), axis=-1)

    rel_u = rel / np.maximum(np.linalg.norm(rel, axis=-1, keepdims=True), 1e-9)
    ang_mom = (rel_u[..., 0] * heading[..., 1]
               - rel_u[..., 1] * heading[..., 0]).mean(axis=1)

    out = np.stack([
        speed.mean(axis=1), speed.std(axis=1),
        d_nn1.mean(axis=1), d_nn1.std(axis=1),
        pair.mean(axis=1), pair.min(axis=1), pair.max(axis=1),
        group_radius,
        polarization,
        ang_mom,
        d_wall.mean(axis=1), d_wall.min(axis=1),
        dt,
    ], axis=-1)
    return out.astype(np.float32)                            # [T, G]


def summarize_window(win, feature_names):
    """Collapse one [T, 5, F] window into a vector of group statistics.

    Args:
        win: [T, 5, F] RAW (unscaled) features for one window.
        feature_names: the F feature names, so columns are found by name.

    Returns:
        [len(SUMMARY_NAMES)] float32
    """
    col = {n: i for i, n in enumerate(feature_names)}
    T = win.shape[0]

    x = win[..., col["x"]]                    # [T, 5]
    y = win[..., col["y"]]
    vx = win[..., col["vx"]]
    vy = win[..., col["vy"]]
    speed = win[..., col["speed"]]
    ax = win[..., col["ax"]]
    ay = win[..., col["ay"]]
    dt = win[..., col["dt"]][:, 0]            # shared across agents
    d_wall = win[..., col["d_wall"]]
    sin_th = win[..., col["sin_theta"]]
    cos_th = win[..., col["cos_theta"]]
    d_nn1 = win[..., col["d_nn1"]]

    pos = np.stack([x, y], axis=-1)           # [T, 5, 2]

    # Pairwise distances at each event.
    diff = pos[:, None, :, :] - pos[:, :, None, :]
    dist = np.linalg.norm(diff, axis=-1)      # [T, 5, 5]
    iu = np.triu_indices(pos.shape[1], k=1)
    pair = dist[:, iu[0], iu[1]]              # [T, 10] the 10 unordered pairs

    # Group centroid and radius of gyration at each event.
    centroid = pos.mean(axis=1, keepdims=True)
    group_radius = np.linalg.norm(pos - centroid, axis=-1).mean(axis=1)  # [T]

    # Polarization: how aligned the group is. 1 = all swimming the same way.
    heading = np.stack([cos_th, sin_th], axis=-1)        # [T, 5, 2]
    polarization = np.linalg.norm(heading.mean(axis=1), axis=-1)  # [T]

    # Angular momentum about the centroid, normalized. 1 = perfect milling.
    rel = pos - centroid                                  # [T, 5, 2]
    rel_u = rel / np.maximum(np.linalg.norm(rel, axis=-1, keepdims=True), 1e-9)
    cross = rel_u[..., 0] * heading[..., 1] - rel_u[..., 1] * heading[..., 0]
    ang_mom = cross.mean(axis=1)                          # [T], signed
    milling = np.abs(ang_mom)

    # Heading change per second, an unsigned turning rate.
    dth = np.arctan2(sin_th, cos_th)
    dth = np.diff(dth, axis=0)
    dth = (dth + np.pi) % (2 * np.pi) - np.pi             # wrap to [-pi, pi]
    turn_rate = np.abs(dth) / np.maximum(dt[1:, None], 1e-6)

    # Nearest-neighbor persistence: how often the identity of an agent's
    # nearest neighbor survives from one kick to the next. Low persistence is
    # a signature of a shuffling, weakly cohesive group.
    dist_masked = np.where(np.eye(pos.shape[1], dtype=bool)[None], np.inf, dist)
    nn = np.argmin(dist_masked, axis=-1)                  # [T, 5]
    nn_persistence = float((nn[1:] == nn[:-1]).mean()) if T > 1 else 1.0

    accel_mag = np.sqrt(ax ** 2 + ay ** 2)

    out = [
        speed.mean(), speed.std(),
        # median, not mean: acceleration is a finite difference over an
        # interval that can be 2 ms, so its mean is hostage to a few outliers.
        np.median(accel_mag),
        dt.mean(), dt.std(),
        d_nn1.mean(), d_nn1.std(),
        pair.mean(), pair.min(),
        group_radius.mean(), group_radius.std(),
        polarization.mean(), polarization.std(),
        ang_mom.mean(), milling.mean(),
        d_wall.mean(), d_wall.std(),
        np.median(turn_rate), turn_rate.std(),
        nn_persistence,
    ]
    return np.asarray(out, dtype=np.float32)
