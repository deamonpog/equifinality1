"""Pluggable rule tree generation strategies and deduplication pipeline.

Each generator is a function with signature:
    (n_candidates, max_depth, seed, **kwargs) -> List[Node]

Built-in generators:
    random_pseudo  - standard pseudo-random sampling
    random_quasi   - Halton (low-discrepancy) sequence-driven sampling
    random_mixed   - 50/50 split of pseudo and quasi (default)

To add a new generator (e.g. a GP-based one), define a new function with
the same signature and register it in GENERATORS. The deduplication
pipeline (generate_tree_library) is shared across all generators.
"""

import random
from typing import Callable, Dict, List, Tuple

from genmod.data.rule_trees import (
    FACTOR_NAMES,
    LeafNode,
    Node,
    OPERATORS,
    OpNode,
    tree_to_str,
)


# ---------------------------------------------------------------------------
# Halton sequence (low-discrepancy quasi-random)
# ---------------------------------------------------------------------------

# First few primes for Halton bases
_HALTON_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]


def _halton(index, base):
    """Compute the index-th term of the Halton sequence in the given base."""
    f = 1.0
    r = 0.0
    i = index
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


def _halton_point(index, dim):
    """Return a dim-dimensional Halton point at the given index (1-based)."""
    return [_halton(index, _HALTON_PRIMES[d]) for d in range(dim)]


# ---------------------------------------------------------------------------
# Tree construction primitives
# ---------------------------------------------------------------------------

# Probability of stopping at each level (becoming a leaf instead of an op).
# At max_depth we always stop.
_STOP_PROB = 0.3


def _build_random_tree(max_depth, rng):
    """Pseudo-random tree builder using a python random.Random."""
    if max_depth <= 0 or rng.random() < _STOP_PROB:
        return LeafNode(factor=rng.choice(FACTOR_NAMES))
    op = rng.choice(OPERATORS)
    left = _build_random_tree(max_depth - 1, rng)
    right = _build_random_tree(max_depth - 1, rng)
    return OpNode(op=op, left=left, right=right)


def _build_quasi_tree(max_depth, halton_point, cursor):
    """Quasi-random tree builder driven by a Halton point.

    halton_point is a fixed sequence of values in [0, 1). cursor is a
    mutable list with one element (an int index) so we can advance through
    the point as we recurse. If we run out of dimensions, fall back to
    cycling through the point.
    """
    def next_val():
        idx = cursor[0] % len(halton_point)
        cursor[0] += 1
        return halton_point[idx]

    def build(depth):
        stop = (depth <= 0) or (next_val() < _STOP_PROB)
        if stop:
            factor_idx = int(next_val() * len(FACTOR_NAMES)) % len(FACTOR_NAMES)
            return LeafNode(factor=FACTOR_NAMES[factor_idx])
        op_idx = int(next_val() * len(OPERATORS)) % len(OPERATORS)
        left = build(depth - 1)
        right = build(depth - 1)
        return OpNode(op=OPERATORS[op_idx], left=left, right=right)

    return build(max_depth)


# ---------------------------------------------------------------------------
# Generator implementations
# ---------------------------------------------------------------------------

def random_pseudo(n_candidates, max_depth, seed, **kwargs):
    """Pseudo-random tree generation using Python's random module."""
    rng = random.Random(seed)
    return [_build_random_tree(max_depth, rng) for _ in range(n_candidates)]


def random_quasi(n_candidates, max_depth, seed, **kwargs):
    """Quasi-random tree generation using Halton sequences for
    low-discrepancy coverage of the (factor, operator, depth) decision space.
    """
    # Each tree consumes a few decisions; budget enough Halton dimensions.
    # A depth-d binary tree has up to 2^(d+1)-1 nodes, each needing ~2 decisions.
    dim = max(4, 2 ** (max_depth + 1))
    if dim > len(_HALTON_PRIMES):
        dim = len(_HALTON_PRIMES)

    trees = []
    for i in range(n_candidates):
        # Offset by seed so different seeds give different sub-sequences
        point = _halton_point(i + 1 + seed * 1000, dim)
        cursor = [0]
        trees.append(_build_quasi_tree(max_depth, point, cursor))
    return trees


def random_mixed(n_candidates, max_depth, seed, **kwargs):
    """50/50 mix of pseudo-random and quasi-random generation."""
    half = n_candidates // 2
    pseudo_trees = random_pseudo(half, max_depth, seed)
    quasi_trees = random_quasi(n_candidates - half, max_depth, seed + 1)
    return pseudo_trees + quasi_trees


# Registry of generators (extensible)
GENERATORS: Dict[str, Callable] = {
    "pseudo": random_pseudo,
    "quasi": random_quasi,
    "mixed": random_mixed,
}


# ---------------------------------------------------------------------------
# Behavioral fingerprinting for deduplication
# ---------------------------------------------------------------------------

def _coarse_grid_hash(grid, macro=4):
    """Hash a grid by counting type-A fraction in macro x macro blocks,
    quantized to 10 bins. Returns a hashable tuple."""
    size = len(grid)
    block = max(size // macro, 1)
    bins = []
    for br in range(0, size, block):
        for bc in range(0, size, block):
            count_a, count_total = 0, 0
            for r in range(br, min(br + block, size)):
                for c in range(bc, min(bc + block, size)):
                    if grid[r][c] != 0:
                        count_total += 1
                        if grid[r][c] == 1:
                            count_a += 1
            if count_total == 0:
                bins.append(5)
            else:
                bins.append(int(count_a / count_total * 10))
    return tuple(bins)


# Probe configurations for deduplication. Each probe is a small simulation
# (100 steps) on a fixed seed. Three probes give robust collision avoidance.
DEFAULT_PROBES = [
    {"grid_size": 50, "occupancy": 0.95, "seed": 1001, "max_steps": 100},
    {"grid_size": 50, "occupancy": 0.95, "seed": 2002, "max_steps": 100},
    {"grid_size": 50, "occupancy": 0.95, "seed": 3003, "max_steps": 100},
]


def fingerprint_tree(tree, probe_configs=None):
    """Run the tree on probe scenarios and return a behavioral fingerprint.

    Imported lazily to avoid circular dependency with the simulator module.
    """
    from genmod.data.schelling_ruletree import run_ruletree

    if probe_configs is None:
        probe_configs = DEFAULT_PROBES

    parts = []
    for pc in probe_configs:
        result = run_ruletree(
            grid_size=pc["grid_size"],
            tree=tree,
            occupancy=pc["occupancy"],
            max_steps=pc["max_steps"],
            snapshot_ticks=[pc["max_steps"]],  # only need final state
            seed=pc["seed"],
        )
        final_grid = result["snapshots"][-1]
        gh = _coarse_grid_hash(final_grid)
        total_moves = result.get("total_moves", 0)
        eq_step = result["steps_to_equilibrium"]
        parts.append((gh, total_moves, eq_step))
    return tuple(parts)


def _fingerprint_worker(args):
    """Multiprocessing worker: compute fingerprint for one (key, tree).

    Returns (key, fingerprint). Defined at module top level so Pool can
    pickle it.
    """
    key, tree, probe_configs = args
    fp = fingerprint_tree(tree, probe_configs)
    return key, fp


# ---------------------------------------------------------------------------
# Main entry point: generate + structural dedup + behavioral dedup
# ---------------------------------------------------------------------------

def generate_tree_library(
    n_candidates: int = 10000,
    max_depth: int = 3,
    seed: int = 42,
    method: str = "mixed",
    probe_configs=None,
    n_workers: int = 1,
) -> List[Tuple[int, Node, str]]:
    """Generate, deduplicate, and label a library of rule trees.

    Pipeline:
      1. Generate n_candidates trees using the chosen method
      2. Structural dedup: drop trees with identical canonical strings
      3. Behavioral dedup: drop trees with identical behavioral fingerprints
      4. Sort by tree string for reproducible labels

    n_workers: number of multiprocessing workers for the (CPU-bound)
    fingerprinting step. n_workers=1 uses a serial loop. The output is
    bit-identical regardless of n_workers because each fingerprint is
    deterministic in its tree and probe seeds.

    Returns: list of (label, Node, tree_str) tuples.
    """
    if method not in GENERATORS:
        raise ValueError("Unknown method '{}'. Choose from {}".format(
            method, list(GENERATORS)))

    print("Generating {} candidate trees with method '{}' (max_depth={})...".format(
        n_candidates, method, max_depth))
    candidates = GENERATORS[method](n_candidates, max_depth, seed)

    # Structural dedup
    by_str = {}
    for t in candidates:
        s = tree_to_str(t)
        if s not in by_str:
            by_str[s] = t
    print("  {} structurally unique trees".format(len(by_str)))

    # Behavioral dedup
    items = sorted(by_str.items(), key=lambda kv: kv[0])
    print("  Running behavioral fingerprinting on {} trees ({} workers)...".format(
        len(items), n_workers))
    fingerprints = {}
    n_collisions = 0

    if n_workers <= 1:
        # Serial path
        for i, (s, t) in enumerate(items):
            if (i + 1) % 100 == 0:
                print("    fingerprinted {}/{} (kept {})".format(
                    i + 1, len(items), len(fingerprints)))
            fp = fingerprint_tree(t, probe_configs)
            if fp not in fingerprints:
                fingerprints[fp] = (s, t)
            else:
                n_collisions += 1
    else:
        # Parallel path: dispatch one task per candidate. To keep the
        # deduplication deterministic (independent of worker completion
        # order), we collect all (key, fp) pairs first, then apply them
        # in the original sorted-by-key order so the same representative
        # always wins for any given fingerprint cluster.
        from multiprocessing import Pool
        args_iter = [(s, t, probe_configs) for s, t in items]
        chunksize = max(1, len(args_iter) // (n_workers * 8))
        key_to_fp = {}
        completed = 0
        with Pool(n_workers) as pool:
            for key, fp in pool.imap_unordered(
                _fingerprint_worker, args_iter, chunksize=chunksize
            ):
                key_to_fp[key] = fp
                completed += 1
                if completed % 200 == 0:
                    print("    fingerprinted {}/{}".format(completed, len(items)))

        # Apply dedup in deterministic key order
        for s, t in items:
            fp = key_to_fp[s]
            if fp not in fingerprints:
                fingerprints[fp] = (s, t)
            else:
                n_collisions += 1

    print("  {} behaviorally unique trees ({} collisions)".format(
        len(fingerprints), n_collisions))

    # Sort by tree string for reproducible labels
    unique_items = sorted(fingerprints.values(), key=lambda x: x[0])
    library = [(label, t, s) for label, (s, t) in enumerate(unique_items)]
    return library
