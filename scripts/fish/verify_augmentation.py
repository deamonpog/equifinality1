"""Verify that the feature-space symmetries really are symmetries.

genmod/fish/augment.py rotates and reflects windows by transforming the
FEATURE VECTOR directly, which is fast but only correct if the transformation
it applies is exactly what you would get by transforming the raw POSITIONS
and recomputing every feature from scratch.

This script checks that equivalence on real runs:

    rotate(features(positions), phi)  ==  features(rotate(positions, phi))

If these disagree, the model is being taught a symmetry the data does not
have, and no training curve would reveal it.

Run from the repository root:
    python scripts/fish/verify_augmentation.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

from genmod.fish.augment import INVARIANT, permute_agents, reflect, rotate
from genmod.fish.features import FEATURE_NAMES, compute_features
from genmod.fish.loader import ARENA_RADIUS, load_rule_lookup

DATA_DIR = os.path.join("FISH_DATA", "Agents")


def load_one_run(source_file, exp_id):
    raw = np.loadtxt(os.path.join(DATA_DIR, source_file))
    m = raw[:, 0].astype(int) == exp_id
    agent = raw[m, 1].astype(int)
    t = raw[m & True, 2][agent == 1]
    pos = np.stack([raw[m][agent == a][:, 3:5] for a in range(1, 6)], axis=1)
    return t, pos


def rotate_positions(pos, phi):
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c, -s], [s, c]])
    return pos @ R.T


def reflect_positions(pos):
    out = pos.copy()
    out[..., 1] *= -1.0
    return out


def main():
    lookup = load_rule_lookup()
    rng = np.random.default_rng(0)

    print("Checking rotate/reflect equivalence on real runs")
    print("  claim: transforming the FEATURES equals recomputing the features")
    print("         from transformed POSITIONS\n")

    worst_rot = 0.0
    worst_ref = 0.0

    # A few runs spanning different rules, and several rotation angles.
    for entry in [lookup[0], lookup[3], lookup[6], lookup[10]]:
        t, pos = load_one_run(entry["source_file"], exp_id=1)
        feats, _ = compute_features(t, pos, radius=ARENA_RADIUS)

        for phi in rng.uniform(0, 2 * np.pi, size=3):
            # Path A: recompute features from rotated positions.
            f_recomputed, _ = compute_features(
                t, rotate_positions(pos, phi), radius=ARENA_RADIUS)
            # Path B: rotate the features directly.
            f_transformed = rotate(feats, phi)
            err = np.abs(f_recomputed - f_transformed).max()
            worst_rot = max(worst_rot, float(err))

        f_recomputed, _ = compute_features(
            t, reflect_positions(pos), radius=ARENA_RADIUS)
        f_transformed = reflect(feats)
        err = np.abs(f_recomputed - f_transformed).max()
        worst_ref = max(worst_ref, float(err))

        print("  rule %-2d %-32s  rot err %.2e   refl err %.2e"
              % (entry["rule_id"], entry["rule_name"], worst_rot, worst_ref))

    tol = 1e-4  # float32 storage, and angles reconstructed through atan2
    print("\n[1] Rotation")
    print("    max |recomputed - transformed| = %.3e   passes = %s"
          % (worst_rot, worst_rot < tol))
    print("[2] Reflection")
    print("    max |recomputed - transformed| = %.3e   passes = %s"
          % (worst_ref, worst_ref < tol))

    # ---- invariants really invariant? ---------------------------------
    t, pos = load_one_run(lookup[5]["source_file"], exp_id=2)
    feats, _ = compute_features(t, pos, radius=ARENA_RADIUS)
    rotated = rotate(feats, 1.234)
    reflected = reflect(feats)

    print("\n[3] Features that must NOT change under rotation or reflection")
    ok = True
    for name in INVARIANT:
        j = FEATURE_NAMES.index(name)
        d_rot = float(np.abs(rotated[..., j] - feats[..., j]).max())
        d_ref = float(np.abs(reflected[..., j] - feats[..., j]).max())
        good = d_rot < 1e-6 and d_ref < 1e-6
        ok &= good
        print("    %-16s rot %.1e  refl %.1e  %s"
              % (name, d_rot, d_ref, "ok" if good else "CHANGED"))
    print("    all invariants hold ... %s" % ok)

    # ---- permutation should leave group statistics untouched -----------
    print("\n[4] Agent permutation is a relabelling, not a change of state")
    permuted = permute_agents(feats, np.random.default_rng(7))
    same_multiset = np.allclose(np.sort(feats, axis=1), np.sort(permuted, axis=1))
    print("    the set of agent states is unchanged ... %s" % same_multiset)

    print("\nAll checks %s" % ("PASSED" if (worst_rot < tol and worst_ref < tol
                                            and ok and same_multiset)
                               else "FAILED"))


if __name__ == "__main__":
    main()
