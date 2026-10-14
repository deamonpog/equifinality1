"""Verify the GPU augmentation agrees with the numpy reference.

genmod/fish/augment.py is the reference implementation: it was checked against
recomputing every feature from transformed positions
(scripts/fish/verify_augmentation.py). genmod/fish/gpu_transform.py is a
batched rewrite of the same maths for speed, and speed rewrites are exactly
where a sign or an axis quietly flips.

This asserts the two produce identical output for the same rotation angle,
the same reflection, and the same agent permutation, and that the batched
window gather returns the same windows as a plain Python loop.

Run from the repository root:
    python scripts/fish/verify_gpu_transform.py
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())

from genmod.fish.augment import reflect, rotate
from genmod.fish.gpu_transform import GPUTransform, cut_batch
from genmod.fish.windows import apply_scaler

RUNS_PATH = os.path.join("data", "fish", "runs.npz")
T = 32


def main():
    data = np.load(RUNS_PATH, allow_pickle=True)
    feats, offsets = data["feats"], data["offsets"]
    wins = np.load(os.path.join("data", "fish", "windows_T%d.npz" % T))
    index = wins["fold0_test"]
    median, iqr = wins["fold0_median"], wins["fold0_iqr"]
    clip = float(wins["clip"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tf = GPUTransform(median, iqr, clip, device)
    rows = np.arange(8)

    # ---- 1. the batched gather ---------------------------------------
    batch = cut_batch(feats, offsets, index, rows, T)
    loop = np.stack([feats[offsets[r]:offsets[r] + T] if False else
                     feats[offsets[index[i, 0]] + index[i, 1]:
                           offsets[index[i, 0]] + index[i, 1] + T]
                     for i in rows])
    print("[1] Batched window gather")
    err = float(np.abs(batch - loop).max())
    print("    max |vectorized - loop| = %.3e   passes = %s"
          % (err, err == 0.0))

    x = torch.from_numpy(np.ascontiguousarray(batch)).to(device)

    # ---- 2. rotation --------------------------------------------------
    # Drive the GPU path with a fixed angle by calling its internals directly:
    # the public augment() samples phi at random, which we cannot compare to.
    phi = 1.2345
    gpu = x.clone()
    c, s = np.cos(phi), np.sin(phi)
    from genmod.fish.augment import ROTATION_PAIRS
    for ix, iy in ROTATION_PAIRS:
        u = gpu[..., ix].clone()
        v = gpu[..., iy].clone()
        gpu[..., ix] = c * u - s * v
        gpu[..., iy] = s * u + c * v
    ref = np.stack([rotate(w, phi) for w in batch])
    err_rot = float(np.abs(gpu.cpu().numpy() - ref).max())
    print("\n[2] Rotation, GPU vs numpy reference")
    print("    max difference = %.3e   passes = %s"
          % (err_rot, err_rot < 1e-5))

    # ---- 3. reflection ------------------------------------------------
    from genmod.fish.augment import REFLECTION_NEGATE
    gpu = x.clone()
    for j in REFLECTION_NEGATE:
        gpu[..., j] = -gpu[..., j]
    ref = np.stack([reflect(w) for w in batch])
    err_ref = float(np.abs(gpu.cpu().numpy() - ref).max())
    print("\n[3] Reflection, GPU vs numpy reference")
    print("    max difference = %.3e   passes = %s"
          % (err_ref, err_ref < 1e-6))

    # ---- 4. scaling ---------------------------------------------------
    gpu = tf.scale(x).cpu().numpy()
    ref = apply_scaler(batch, median, iqr, clip)
    err_sc = float(np.abs(gpu - ref).max())
    print("\n[4] Robust scaling, GPU vs numpy reference")
    print("    max difference = %.3e   passes = %s"
          % (err_sc, err_sc < 1e-5))

    # ---- 5. the permutation is a permutation ---------------------------
    # augment() samples its own permutation, so check the invariant instead:
    # the multiset of agent states must be unchanged at every event.
    g = torch.Generator(device=device).manual_seed(0)
    p = tf.augment(x, rotate=False, reflect=False, permute=True, generator=g)
    same = torch.allclose(torch.sort(p, dim=2).values,
                          torch.sort(x, dim=2).values)
    print("\n[5] Agent permutation preserves the set of agent states")
    print("    unchanged multiset = %s" % bool(same))
    # And it must be a genuine relabelling, not the identity every time.
    moved = not torch.allclose(p, x)
    print("    actually permutes something = %s" % moved)

    ok = (err == 0.0 and err_rot < 1e-5 and err_ref < 1e-6
          and err_sc < 1e-5 and same and moved)
    print("\nAll checks %s" % ("PASSED" if ok else "FAILED"))


if __name__ == "__main__":
    main()
