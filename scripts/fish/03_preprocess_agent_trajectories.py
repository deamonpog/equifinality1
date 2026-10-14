"""Preprocess the simulated-agent trajectories into a feature tensor.

Loads all 550 runs, computes per-(agent, event) features, applies the burn-in,
verifies the result, and writes one file that every later step reads:

    data/fish/runs.npz
        feats        [E, 5, F] float32, all runs concatenated along events
        offsets      [n_runs + 1] int64, run i occupies feats[o[i]:o[i+1]]
        times        [E] float32, kick time of each event
        rule_id      [n_runs] int64
        family_id    [n_runs] int64
        neighbor_count [n_runs] int64
        exp_id       [n_runs] int64
        feature_names, n_basic, arena_radius, burn_in

Runs have different lengths, so they are stored concatenated with an offset
index rather than padded into a rectangular array.

Run from the repository root:
    python scripts/fish/03_preprocess_agent_trajectories.py
"""

import os

import numpy as np

import sys
sys.path.insert(0, os.getcwd())

from genmod.fish.features import (BURN_IN_EVENTS, FEATURE_NAMES, N_BASIC,
                                  compute_features)
from genmod.fish.loader import ARENA_RADIUS, load_runs

OUT_PATH = os.path.join("data", "fish", "runs.npz")


def main():
    print("Loading raw runs ...")
    runs = load_runs()
    print("  %d runs, %d rules, %d experiments per rule"
          % (len(runs), len({r.rule_id for r in runs}),
             len({r.exp_id for r in runs})))

    print("\nComputing features (burn-in = %d events) ..." % BURN_IN_EVENTS)
    feats_list, times_list, offsets = [], [], [0]
    for run in runs:
        f, t = compute_features(run.time, run.pos, radius=ARENA_RADIUS)
        feats_list.append(f)
        times_list.append(t.astype(np.float32))
        offsets.append(offsets[-1] + f.shape[0])

    feats = np.concatenate(feats_list, axis=0)
    times = np.concatenate(times_list, axis=0)
    offsets = np.asarray(offsets, dtype=np.int64)

    lengths = np.diff(offsets)
    print("  feature tensor: %s  (%.0f MB)"
          % (feats.shape, feats.nbytes / 1e6))
    print("  events per run: min %d  median %d  max %d"
          % (lengths.min(), int(np.median(lengths)), lengths.max()))

    # ---- check 1: round trip -----------------------------------------
    # Features 0 and 1 are x, y normalized by R. Undo that and compare with
    # the raw positions the run object still holds. They must match exactly
    # (up to float32 storage), or the alignment between the derivative grid
    # and the burn-in is wrong.
    print("\n[1] Round-trip check (features -> positions -> raw .dat)")
    worst = 0.0
    for i, run in enumerate(runs):
        f = feats[offsets[i]:offsets[i + 1]]
        recon = f[..., :2].astype(np.float64) * ARENA_RADIUS
        # compute_features drops 2 events to derivatives, then burn_in more.
        raw = run.pos[2 + BURN_IN_EVENTS:]
        worst = max(worst, float(np.abs(recon - raw).max()))
    print("    max |reconstructed - raw| = %.3e m" % worst)
    print("    passes (< 1e-6 m) ......... %s" % (worst < 1e-6))

    # ---- check 2: arena containment ----------------------------------
    print("\n[2] Arena containment after burn-in")
    d_wall = feats[..., FEATURE_NAMES.index("d_wall")]
    print("    d_wall min ................ %.4f" % d_wall.min())
    print("    d_wall max ................ %.4f" % d_wall.max())
    n_out = int((d_wall < 0).sum())
    print("    events outside the wall ... %d" % n_out)
    print("    burn-in removed the startup transient ... %s" % (n_out == 0))

    # ---- check 3: finiteness and scale --------------------------------
    print("\n[3] Feature health")
    print("    non-finite values ......... %d" % int((~np.isfinite(feats)).sum()))
    print("    %-16s %9s %9s %9s %9s" % ("feature", "min", "mean", "std", "max"))
    for j, name in enumerate(FEATURE_NAMES):
        col = feats[..., j]
        tag = "basic" if j < N_BASIC else "rel"
        print("    %-16s %9.3f %9.3f %9.3f %9.3f   (%s)"
              % (name, col.min(), col.mean(), col.std(), col.max(), tag))

    # ---- check 4: does the signal exist at all? -----------------------
    # Before training anything, look at whether the rules separate on a
    # single obvious statistic. If nearest-1 and all-4 have identical mean
    # neighbor distance, we would want to know now, not after step 09.
    print("\n[4] Sanity: mean nearest-neighbor distance by rule")
    j = FEATURE_NAMES.index("d_nn1")
    rule_ids = np.array([r.rule_id for r in runs])
    print("    rule  mean d_nn1   (arena radii)")
    for rid in range(11):
        sel = np.where(rule_ids == rid)[0]
        vals = np.concatenate([feats[offsets[i]:offsets[i + 1], :, j].ravel()
                               for i in sel])
        bar = "#" * int(round(vals.mean() * 120))
        print("    %4d  %.4f  %s" % (rid, vals.mean(), bar))
    print("    (a spread here means the rules are at least partly separable;")
    print("     overlapping values are exactly what conformal sets are for)")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        feats=feats,
        offsets=offsets,
        times=times,
        rule_id=np.array([r.rule_id for r in runs], dtype=np.int64),
        family_id=np.array([r.family_id for r in runs], dtype=np.int64),
        neighbor_count=np.array([r.neighbor_count for r in runs], dtype=np.int64),
        exp_id=np.array([r.exp_id for r in runs], dtype=np.int64),
        feature_names=np.array(FEATURE_NAMES),
        n_basic=np.int64(N_BASIC),
        arena_radius=np.float64(ARENA_RADIUS),
        burn_in=np.int64(BURN_IN_EVENTS),
    )
    print("\nWrote %s (%.0f MB on disk)"
          % (OUT_PATH, os.path.getsize(OUT_PATH) / 1e6))


if __name__ == "__main__":
    main()
