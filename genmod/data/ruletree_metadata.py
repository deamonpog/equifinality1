"""Metadata helpers for grouping and analyzing rule trees.

Provides coarse groupings (analogous to Wolfram classes for ECA or
segregation levels for threshold Schelling). Trees are grouped by their
dominant factor -- the leaf factor that appears most often.
"""

from genmod.data.rule_trees import (
    FACTOR_NAMES,
    LeafNode,
)


def _collect_leaves(node):
    """Collect all leaf factor names appearing in the tree."""
    if isinstance(node, LeafNode):
        return [node.factor]
    leaves = []
    leaves.extend(_collect_leaves(node.left))
    leaves.extend(_collect_leaves(node.right))
    return leaves


def get_dominant_factor(tree):
    """Return the most frequently appearing leaf factor in the tree.

    Ties are broken alphabetically for reproducibility.
    """
    leaves = _collect_leaves(tree)
    if not leaves:
        return FACTOR_NAMES[0]
    counts = {}
    for f in leaves:
        counts[f] = counts.get(f, 0) + 1
    max_count = max(counts.values())
    candidates = sorted(f for f, c in counts.items() if c == max_count)
    return candidates[0]


def get_tree_complexity(tree):
    """Return the total number of nodes in the tree."""
    if isinstance(tree, LeafNode):
        return 1
    return 1 + get_tree_complexity(tree.left) + get_tree_complexity(tree.right)


def get_tree_group(tree):
    """Return group index (0-5) based on the dominant factor.

    Group ordering matches FACTOR_NAMES:
        0=race, 1=age, 2=dist, 3=isol, 4=move, 5=neigh
    """
    return FACTOR_NAMES.index(get_dominant_factor(tree))


def get_group_name(group_idx):
    """Human-readable name for a tree group."""
    if 0 <= group_idx < len(FACTOR_NAMES):
        return FACTOR_NAMES[group_idx].capitalize()
    return "Unknown"


def factors_in_tree(tree):
    """Return the sorted list of unique factors used by the tree."""
    return sorted(set(_collect_leaves(tree)))
