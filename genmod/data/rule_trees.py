"""Rule tree data structures, factor definitions, and evaluation.

A rule tree is a small expression tree built from 6 primitive factors
combined with arithmetic operators {+, -, *, /}. Each tree defines an
agent utility function u(agent, location) for the Schelling model.

Factor definitions match Gunaratne et al. (2023), JASSS 26(2):

  F_Race(a, i)  - fraction of same-type neighbors at i
  F_Age(a, i)   - mean tenure (steps occupied) of neighbors at i
  F_Dist(a, i)  - Euclidean distance from agent's home to i, normalized
  F_Isol(a, i)  - fraction of vacant cells in Moore neighborhood of i
  F_Move(a, i)  - asymmetric: agent's recent relocation rate
  F_Neigh(a, i) - mean stored utility of neighbors at i (two-pass)

This module contains tree structures, factor functions, evaluation, and
serialization. Tree generation lives in tree_generators.py.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union


# ---------------------------------------------------------------------------
# Tree data structures
# ---------------------------------------------------------------------------

FACTOR_NAMES = ["race", "age", "dist", "isol", "move", "neigh"]
OPERATORS = ["+", "-", "*", "/"]


@dataclass(frozen=True)
class LeafNode:
    """Terminal node referencing one of the 6 primitive factors."""
    factor: str


@dataclass(frozen=True)
class OpNode:
    """Internal node combining two subtrees with an arithmetic operator."""
    op: str
    left: "Node"
    right: "Node"


Node = Union[LeafNode, OpNode]


# ---------------------------------------------------------------------------
# Factor context: state bundle passed to factor functions
# ---------------------------------------------------------------------------

@dataclass
class FactorContext:
    """All state needed to evaluate a factor at one (agent, location) pair.

    The simulator constructs this once per (agent, candidate location) and
    passes it to evaluate_tree.
    """
    grid: List[List[int]]            # 0=vacant, 1=type A, 2=type B
    grid_size: int
    agent_type: int                  # 1 or 2
    eval_row: int                    # candidate location row
    eval_col: int                    # candidate location col
    home_row: int                    # agent's current row
    home_col: int                    # agent's current col
    ages: List[List[int]]            # tenure at each cell (0 if vacant)
    move_history: List[List[List[int]]]  # per-cell list of recent move ticks
    utilities: List[List[float]]     # stored prev-tick utility per cell
    current_step: int                # for f_move recency window
    max_steps: int                   # for normalization


# ---------------------------------------------------------------------------
# Moore neighborhood helper (torus / wrap-around boundary)
# ---------------------------------------------------------------------------

def moore_neighbors(grid, row, col):
    """Return list of (value, nr, nc) for the 8 Moore neighbors with torus
    boundary. Includes both occupied and vacant cells.
    """
    size = len(grid)
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr = (row + dr) % size
            nc = (col + dc) % size
            out.append((grid[nr][nc], nr, nc))
    return out


# ---------------------------------------------------------------------------
# Six primitive factors (per Gunaratne et al. 2023)
# Each returns a float; most are in [0, 1].
# ---------------------------------------------------------------------------

def f_race(ctx):
    """Fraction of same-type neighbors at the evaluation location."""
    nbrs = moore_neighbors(ctx.grid, ctx.eval_row, ctx.eval_col)
    occupied = [v for v, _, _ in nbrs if v != 0]
    if not occupied:
        return 1.0
    return sum(1 for v in occupied if v == ctx.agent_type) / len(occupied)


def f_age(ctx):
    """Mean tenure of occupied neighbors at the evaluation location,
    normalized by max_steps."""
    nbrs = moore_neighbors(ctx.grid, ctx.eval_row, ctx.eval_col)
    occupied = [(r, c) for v, r, c in nbrs if v != 0]
    if not occupied:
        return 0.0
    total = sum(ctx.ages[r][c] for r, c in occupied)
    mean_age = total / len(occupied)
    norm = max(ctx.max_steps, 1)
    return min(mean_age / norm, 1.0)


def f_dist(ctx):
    """Euclidean distance from agent home to evaluation location, normalized
    by the grid diagonal. Uses torus distance (shortest wrap-around path)."""
    size = ctx.grid_size
    drow = abs(ctx.home_row - ctx.eval_row)
    dcol = abs(ctx.home_col - ctx.eval_col)
    drow = min(drow, size - drow)
    dcol = min(dcol, size - dcol)
    dist = math.sqrt(drow * drow + dcol * dcol)
    max_d = math.sqrt(2.0) * (size / 2.0)
    if max_d == 0:
        return 0.0
    return min(dist / max_d, 1.0)


def f_isol(ctx):
    """Fraction of vacant cells in the Moore neighborhood of the eval
    location."""
    nbrs = moore_neighbors(ctx.grid, ctx.eval_row, ctx.eval_col)
    if not nbrs:
        return 0.0
    return sum(1 for v, _, _ in nbrs if v == 0) / len(nbrs)


# Recency window for f_move (per Gunaratne paper: last 10 ticks)
F_MOVE_WINDOW = 10


def f_move(ctx):
    """Asymmetric recent-relocation rate.

    If the eval location is the agent's home, return 1 - (recent moves / 10).
    Otherwise return (recent moves / 10). This penalizes moving when the
    agent has been settled and rewards moving when the agent has been mobile.
    """
    moves = ctx.move_history[ctx.home_row][ctx.home_col]
    cutoff = ctx.current_step - F_MOVE_WINDOW
    recent = sum(1 for t in moves if t > cutoff)
    rate = min(recent / F_MOVE_WINDOW, 1.0)
    is_home = (ctx.eval_row == ctx.home_row and ctx.eval_col == ctx.home_col)
    if is_home:
        return 1.0 - rate
    return rate


def f_neigh(ctx):
    """Mean stored utility of occupied neighbors at the eval location.

    Reads ctx.utilities, which the simulator populates in pass 1 of each
    step (before any movement decisions are made). Avoids recursion.
    """
    nbrs = moore_neighbors(ctx.grid, ctx.eval_row, ctx.eval_col)
    occupied = [(r, c) for v, r, c in nbrs if v != 0]
    if not occupied:
        return 1.0
    total = sum(ctx.utilities[r][c] for r, c in occupied)
    return total / len(occupied)


FACTOR_FUNCTIONS = {
    "race": f_race,
    "age": f_age,
    "dist": f_dist,
    "isol": f_isol,
    "move": f_move,
    "neigh": f_neigh,
}


# ---------------------------------------------------------------------------
# Tree evaluation
# ---------------------------------------------------------------------------

def protected_div(a, b):
    """Division that returns 0.0 when the denominator is near zero."""
    if abs(b) < 1e-10:
        return 0.0
    return a / b


_OP_FUNCS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": protected_div,
}


def evaluate_tree(node, ctx):
    """Recursively evaluate a rule tree given a FactorContext."""
    if isinstance(node, LeafNode):
        return FACTOR_FUNCTIONS[node.factor](ctx)
    left_val = evaluate_tree(node.left, ctx)
    right_val = evaluate_tree(node.right, ctx)
    return _OP_FUNCS[node.op](left_val, right_val)


# ---------------------------------------------------------------------------
# Tree string representation (canonical infix)
# ---------------------------------------------------------------------------

def tree_to_str(node):
    """Convert tree to canonical infix string for display and structural
    deduplication."""
    if isinstance(node, LeafNode):
        return node.factor
    left_s = tree_to_str(node.left)
    right_s = tree_to_str(node.right)
    return "({} {} {})".format(left_s, node.op, right_s)


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------

def tree_to_dict(node):
    """Serialize a tree to a JSON-compatible dict."""
    if isinstance(node, LeafNode):
        return {"type": "leaf", "factor": node.factor}
    return {
        "type": "op",
        "op": node.op,
        "left": tree_to_dict(node.left),
        "right": tree_to_dict(node.right),
    }


def dict_to_tree(d):
    """Deserialize a tree from a dict."""
    if d["type"] == "leaf":
        return LeafNode(factor=d["factor"])
    return OpNode(
        op=d["op"],
        left=dict_to_tree(d["left"]),
        right=dict_to_tree(d["right"]),
    )


# ---------------------------------------------------------------------------
# Tree library save / load
# ---------------------------------------------------------------------------

def save_tree_library(library, path):
    """Save a tree library to JSON.

    library: list of (label, Node, tree_str) tuples.
    """
    entries = []
    for label, tree, tree_str in library:
        entries.append({
            "label": label,
            "tree_str": tree_str,
            "tree": tree_to_dict(tree),
        })
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(entries, f, indent=2)
    print("  Saved {} trees to {}".format(len(entries), p))


def load_tree_library(path):
    """Load a tree library from JSON.

    Returns: list of (label, Node, tree_str) tuples.
    """
    with open(path) as f:
        entries = json.load(f)
    library = []
    for e in entries:
        tree = dict_to_tree(e["tree"])
        library.append((e["label"], tree, e["tree_str"]))
    return library
