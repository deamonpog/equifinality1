"""Verify that the trajectory Transformer is EXACTLY permutation invariant.

The claim in genmod/models/traj_transformer.py is architectural, not learned:
because tokens carry a time embedding but no agent-id embedding, permuting the
five agents permutes the token set, and self-attention is a set operation, so
the CLS output cannot change:

    f(W) = f(PW)   for every permutation P and EVERY choice of weights

That last part is what makes this test strong. If it holds for randomly
initialized weights, it holds for trained weights too, and the permutation
robustness reported in step 11 is a property of the architecture rather than
something the model had to learn from augmentation.

If this test fails, an agent-dependent term has crept into the model, and the
random-permutation augmentation in training would be silently papering over
it.

Run from the repository root:
    python scripts/fish/verify_permutation_invariance.py
"""

import os
import sys

import torch

sys.path.insert(0, os.getcwd())

from genmod.models.traj_transformer import TrajectoryTransformer

T, A, F, C = 16, 5, 20, 11


def main():
    torch.manual_seed(0)
    model = TrajectoryTransformer(
        n_features=F, n_agents=A, max_events=T, num_classes=C,
        d_model=64, n_heads=4, n_layers=3, dropout=0.0,
    ).eval()

    x = torch.randn(4, T, A, F)

    with torch.no_grad():
        base = model(x)

        print("Permuting the five agents and re-running the model")
        print("  (dropout is off; any difference is a real asymmetry)\n")
        worst = 0.0
        for trial in range(10):
            perm = torch.randperm(A)
            # The permutation is applied consistently across all events, which
            # is the physical symmetry: agent identities are relabelled once,
            # not shuffled independently at each timestep.
            out = model(x[:, :, perm, :])
            err = float((out - base).abs().max())
            worst = max(worst, err)
            print("  perm %-18s max |f(PW) - f(W)| = %.3e"
                  % (str(perm.tolist()), err))

    print("\n[1] Permutation invariance")
    print("    worst deviation over 10 permutations = %.3e" % worst)
    print("    passes (< 1e-4, float32 reduction order) ... %s" % (worst < 1e-4))

    # A control: if the model WERE sensitive to agent order, shuffling agents
    # independently at each timestep (which is NOT a symmetry, since it breaks
    # each agent's trajectory into pieces) should change the output. If even
    # that leaves the output identical, the model is ignoring its input.
    with torch.no_grad():
        broken = x.clone()
        for t in range(T):
            broken[:, t] = broken[:, t, torch.randperm(A)]
        ctrl = float((model(broken) - base).abs().max())
    print("\n[2] Control: shuffling agents independently at each event")
    print("    this is NOT a symmetry, so the output SHOULD change")
    print("    max deviation = %.3e   changes = %s" % (ctrl, ctrl > 1e-4))

    ok = worst < 1e-4 and ctrl > 1e-4
    print("\nAll checks %s" % ("PASSED" if ok else "FAILED"))


if __name__ == "__main__":
    main()
