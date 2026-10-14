"""Load the robot trajectories: a physical system with known programmed rules.

The Lei et al. (2020) release also contains trajectories of five CUBOID ROBOTS,
each executing one of the same neighbor-selection strategies in a larger arena.
These are not simulation output. They are real hardware: real actuation error,
real sensing delay, real collisions. But the generating rule is still known
exactly, because each robot was programmed with it.

That makes them the right test of the framework's actual claim. A rule library
learned entirely in simulation is only useful if it identifies mechanisms in a
system that was not simulated.

The domain shift is substantial, and worth stating precisely rather than
hand-waving at:

    property                simulated agents      robots
    arena radius            0.25 m                0.42 m   (1.7x)
    median kick interval    0.434 s               0.920 s  (2.1x slower)
    events per run          ~1600                 140-730
    rules present           11                    10

The robot kick interval is also RULE-DEPENDENT in a way the agents' is not:
the per-file median rises from 0.80 s (all-4) to 1.24 s (nearest-3), while the
agents stay within 0.41-0.44 s. No single robot tau is exact for every rule.

The missing rule is most-influential-k3, which has no robot file. The
classifier still has eleven outputs, so it can emit a class that cannot occur:
an extra way to be wrong, which we do not correct for.

Normalizing positions by the arena radius maps both arenas onto the unit disk,
which is the only reason a transfer is meaningful at all. Everything else --
the timescale, the noise, the physical dynamics -- is genuinely different, and
that is the point.
"""

import os
import re

import numpy as np

from genmod.fish.features import FEATURE_NAMES
from genmod.fish.loader import Run

ROBOT_DIR = os.path.join("FISH_DATA", "Robots")
ROBOT_RADIUS = 0.42  # metres, per the dataset description

# Characteristic kick interval of each domain (median inter-kick time).
# These are the TIME UNITS that make the two domains commensurable.
# Both are the POOLED median of within-run inter-kick intervals, computed by
# agent_tau() and robot_tau() below; the constants are cached values.
TAU_AGENT = 0.434  # s, agent_tau()
TAU_ROBOT = 0.920  # s, robot_tau()

# An earlier, hand-entered value. It equals the median of only the
# no-interaction, nearest-1 and random-1 robot files, not of the whole set.
# Kept for the time-unit sensitivity check (results/fish/robot_tau_sweep).
TAU_ROBOT_LEGACY = 0.840  # s


def median_kick_interval(times):
    """Pooled median inter-kick interval over a list of per-run time arrays."""
    return float(np.median(np.concatenate([np.diff(t) for t in times])))


def robot_tau(data_dir=ROBOT_DIR):
    """Characteristic robot kick interval, from the robot data itself."""
    return median_kick_interval([r.time for r in load_robot_runs(data_dir)])


def agent_tau(path=os.path.join("data", "fish", "runs.npz")):
    """Characteristic agent kick interval, from the preprocessed agent runs."""
    d = np.load(path)
    t, o = d["times"].astype(np.float64), d["offsets"]
    return median_kick_interval([t[o[i]:o[i + 1]] for i in range(len(o) - 1)])


def to_agent_units(feats, tau_src=TAU_ROBOT, tau_dst=TAU_AGENT):
    """Re-express features from one domain's clock in another's.

    Positions are already normalized by arena radius, so every LENGTH in the
    feature vector is dimensionless (arena radii). TIME is not. The features
    carry three time-dependent quantities:

        speed, vx, vy   arena radii per SECOND
        ax, ay          arena radii per SECOND squared
        dt              SECONDS

    Robots kick about 2.1x more slowly than the simulated agents. A robot moving
    identically -- the same fraction of the arena per kick -- therefore registers
    roughly half the speed in radii-per-second. The network would see a
    different animal purely because of the clock, and would be right to be
    confused.

    The physically correct move is to nondimensionalize: adopt each domain's own
    characteristic kick interval tau as its unit of time. Then

        speed * tau     = arena radii traversed PER KICK   (dimensionless)
        accel * tau^2                                      (dimensionless)
        dt / tau                                           (dimensionless, median 1)

    Expressing robot features in agent-equivalent units means multiplying
    velocities by tau_src/tau_dst, accelerations by its square, and dividing dt
    by it. Everything else -- distances, angles, polarization -- is already
    dimensionless and must be left alone.

    This retrains nothing. It corrects the units the data is expressed in
    BEFORE it reaches a network that was fitted in the other unit system.
    """
    r = tau_src / tau_dst
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    out = feats.copy()
    for n in ("vx", "vy", "speed"):
        out[..., idx[n]] *= r
    for n in ("ax", "ay"):
        out[..., idx[n]] *= r ** 2
    out[..., idx["dt"]] /= r
    return out

# (family, k) -> rule id, matching the 11-class agent label space exactly.
# The classifier's outputs mean the same thing in both domains, which is what
# lets us evaluate it on robots without retraining.
RULE_TABLE = [
    ("none", 0), ("nearest", 1), ("nearest", 2), ("nearest", 3),
    ("random", 1), ("random", 2), ("random", 3),
    ("mostinfluential", 1), ("mostinfluential", 2), ("mostinfluential", 3),
    ("all", 4),
]
FAMILY_FROM_FILENAME = {
    "no-interaction": "none",
    "nearest": "nearest",
    "random": "random",
    "mostinfluential": "mostinfluential",
    "all-neighbors": "all",
}


def parse_robot_filename(name):
    """'3.robot-mostinfluential-k2.dat' -> (rule_id, family, k)."""
    m = re.match(r"^\d+\.robot-(.+)-k(\d+)\.dat$", name)
    if m is None:
        raise ValueError("unexpected robot filename: %s" % name)
    family = FAMILY_FROM_FILENAME[m.group(1)]
    k = int(m.group(2))
    return RULE_TABLE.index((family, k)), family, k


def load_robot_runs(data_dir=ROBOT_DIR):
    """Every robot run, in the same Run form as the simulated agents."""
    runs = []
    for name in sorted(f for f in os.listdir(data_dir) if f.endswith(".dat")):
        rule_id, family, k = parse_robot_filename(name)
        raw = np.loadtxt(os.path.join(data_dir, name))
        exp = raw[:, 0].astype(int)
        agent = raw[:, 1].astype(int)

        for e in np.unique(exp):
            m = exp == e
            agents_here = np.unique(agent[m])
            if agents_here.size != 5:
                continue

            t_ref, pos = None, None
            ok = True
            for i, a in enumerate(agents_here):
                sel = m & (agent == a)
                t_a = raw[sel, 2]
                if t_ref is None:
                    t_ref = t_a
                    pos = np.empty((t_ref.shape[0], 5, 2), dtype=np.float64)
                elif t_a.shape != t_ref.shape:
                    # Unlike the simulated agents, a robot run can lose an
                    # individual to a tracking dropout. Skip such runs rather
                    # than interpolate: an invented position is a fabricated
                    # observation, and this is the domain we are testing.
                    ok = False
                    break
                pos[:, i, :] = raw[sel, 3:5]

            if not ok or t_ref.size < 80:
                continue

            runs.append(Run(rule_id=rule_id,
                            family_id=0, neighbor_count=k,
                            exp_id=int(e), time=t_ref, pos=pos))
    return runs
