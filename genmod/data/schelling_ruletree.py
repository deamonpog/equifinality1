"""Schelling segregation simulator with rule-tree utility functions.

Each agent uses a rule tree (built from 6 primitive factors) to compute its
utility at any candidate location. At each step, every agent considers a
random vacant cell and moves there if its rule-tree utility is higher than
the utility at its current location.

This is the simulator backbone for Option D. Parameters match Gunaratne
et al. (2023, JASSS) where applicable:
  - Grid: 50 x 50, torus boundary, Moore neighborhood
  - Density: 0.95
  - Agents: 50/50 of two types
  - Ticks: 500
  - Movement: random vacancy, move if utility there > current

Simplifications versus the paper:
  - No per-agent threshold (the rule tree is the only decision criterion)
  - No random "happy" relocation (m = 0)
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

from genmod.data.rule_trees import (
    FactorContext,
    evaluate_tree,
)


# ---------------------------------------------------------------------------
# Simulation state
# ---------------------------------------------------------------------------

@dataclass
class SimState:
    """Full simulator state.

    grid           - 2D list of cell values (0=vacant, 1=type A, 2=type B)
    ages           - 2D list of tenure (steps occupied at current cell)
    move_history   - 2D list of per-cell lists of recent move ticks
    utilities      - 2D list of stored utilities (populated each step in pass 1)
    grid_size      - integer side length
    current_step   - integer step counter (starts at 0)
    """
    grid: List[List[int]]
    ages: List[List[int]]
    move_history: List[List[List[int]]]
    utilities: List[List[float]]
    grid_size: int
    current_step: int = 0


def create_state(grid_size=50, occupancy=0.95, seed=None):
    """Create the initial random simulation state.

    Returns a SimState with a randomly populated grid (50/50 type ratio)
    and zero-initialized auxiliary grids.
    """
    rng = random.Random(seed)
    n_cells = grid_size * grid_size
    n_occupied = int(round(n_cells * occupancy))
    n_a = n_occupied // 2
    n_b = n_occupied - n_a
    n_vacant = n_cells - n_occupied

    cells = [1] * n_a + [2] * n_b + [0] * n_vacant
    rng.shuffle(cells)

    grid = [cells[i * grid_size:(i + 1) * grid_size] for i in range(grid_size)]
    ages = [[0] * grid_size for _ in range(grid_size)]
    move_history = [[[] for _ in range(grid_size)] for _ in range(grid_size)]
    utilities = [[0.0] * grid_size for _ in range(grid_size)]

    return SimState(
        grid=grid,
        ages=ages,
        move_history=move_history,
        utilities=utilities,
        grid_size=grid_size,
        current_step=0,
    )


# ---------------------------------------------------------------------------
# Utility evaluation helper
# ---------------------------------------------------------------------------

def evaluate_utility(state, tree, agent_type, home_row, home_col,
                     eval_row, eval_col, max_steps):
    """Build a FactorContext and evaluate the rule tree at a candidate."""
    ctx = FactorContext(
        grid=state.grid,
        grid_size=state.grid_size,
        agent_type=agent_type,
        eval_row=eval_row,
        eval_col=eval_col,
        home_row=home_row,
        home_col=home_col,
        ages=state.ages,
        move_history=state.move_history,
        utilities=state.utilities,
        current_step=state.current_step,
        max_steps=max_steps,
    )
    return evaluate_tree(tree, ctx)


# ---------------------------------------------------------------------------
# Single simulation step (two-pass)
# ---------------------------------------------------------------------------

def step_ruletree(state, tree, rng, max_steps):
    """Execute one simulation step.

    Pass 1: compute and store the current-location utility for every
            occupied cell (this populates state.utilities so that f_neigh
            can read neighbor utilities without recursion).
    Pass 2: for each agent (shuffled order), pick a random vacant cell and
            move there if its rule-tree utility exceeds the current utility.

    Returns: number of moves performed during this step.
    """
    size = state.grid_size
    grid = state.grid

    # ---- Pass 1: populate utilities for all occupied cells ----
    # Important: f_neigh reads from state.utilities, which means the values
    # f_neigh sees are from the PREVIOUS step (or initial 0.0 on step 0).
    # That is intentional and matches the paper's two-pass semantics.
    new_utilities = [[0.0] * size for _ in range(size)]
    for r in range(size):
        for c in range(size):
            if grid[r][c] != 0:
                u = evaluate_utility(state, tree, grid[r][c], r, c, r, c,
                                     max_steps)
                new_utilities[r][c] = u
    state.utilities = new_utilities

    # ---- Pass 2: move decisions ----
    agents = [(r, c) for r in range(size) for c in range(size) if grid[r][c] != 0]
    vacancies = [(r, c) for r in range(size) for c in range(size) if grid[r][c] == 0]
    if not vacancies:
        return 0

    rng.shuffle(agents)

    moves = 0
    for ar, ac in agents:
        if grid[ar][ac] == 0:
            # Already moved out earlier in this step
            continue
        agent_type = grid[ar][ac]

        # Pick a random vacancy
        vr, vc = vacancies[rng.randrange(len(vacancies))]

        # Compare utilities
        current_u = state.utilities[ar][ac]
        cand_u = evaluate_utility(state, tree, agent_type, ar, ac, vr, vc,
                                  max_steps)

        if cand_u > current_u:
            # Move agent
            grid[vr][vc] = agent_type
            grid[ar][ac] = 0

            # Reset and update auxiliary state
            state.ages[vr][vc] = 0
            state.ages[ar][ac] = 0

            # Move history travels with the agent
            state.move_history[vr][vc] = list(state.move_history[ar][ac])
            state.move_history[vr][vc].append(state.current_step)
            state.move_history[ar][ac] = []

            # NOTE: do NOT update state.utilities during pass 2. Leaving the
            # pass-1 snapshot intact gives every agent a consistent view of
            # neighbor utilities (the values from before any moves this step).
            # state.utilities will be recomputed at the start of next step.

            # Update vacancy list: remove the destination, add the source
            vacancies.remove((vr, vc))
            vacancies.append((ar, ac))

            moves += 1

    # Increment ages for every occupied cell. Movers had their age reset
    # to 0 above, so they end this step with age = 1 (one step at the new
    # location). Stayers had age N before, so they end with age = N + 1.
    for r in range(size):
        for c in range(size):
            if grid[r][c] != 0:
                state.ages[r][c] += 1

    state.current_step += 1
    return moves


# ---------------------------------------------------------------------------
# Full simulation run
# ---------------------------------------------------------------------------

def run_ruletree(
    grid_size=50,
    tree=None,
    occupancy=0.95,
    max_steps=500,
    snapshot_ticks=None,
    seed=None,
):
    """Run a Schelling rule-tree simulation.

    Args:
        grid_size: square grid side length
        tree: rule tree to use as the utility function
        occupancy: fraction of cells initially occupied
        max_steps: maximum number of simulation steps
        snapshot_ticks: list of tick numbers at which to take snapshots
            (default: [100, 200, 300, 400, 500])
        seed: random seed

    Returns: dict with snapshots, total_moves, steps_to_equilibrium, etc.
    """
    if snapshot_ticks is None:
        snapshot_ticks = [100, 200, 300, 400, 500]

    rng = random.Random(seed)
    state = create_state(grid_size, occupancy, seed=seed)

    snapshot_set = set(snapshot_ticks)
    snapshots = []
    total_moves = 0
    steps_taken = 0

    for step in range(1, max_steps + 1):
        num_moves = step_ruletree(state, tree, rng, max_steps)
        steps_taken = step
        total_moves += num_moves

        if step in snapshot_set:
            snapshots.append([row[:] for row in state.grid])

        if num_moves == 0:
            # Equilibrium reached: stop early. Any snapshot ticks beyond
            # this step will be filled with the equilibrium state below.
            break

    # If we exited early (equilibrium or max_steps), pad snapshots to the
    # expected count so the dataset always sees the same number of frames.
    while len(snapshots) < len(snapshot_ticks):
        snapshots.append([row[:] for row in state.grid])

    return {
        "grid_size": grid_size,
        "snapshots": snapshots,
        "snapshot_ticks": snapshot_ticks,
        "steps_to_equilibrium": steps_taken,
        "total_moves": total_moves,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Data generation helpers
# ---------------------------------------------------------------------------

def generate_ruletree_data(
    tree,
    label,
    grid_size=50,
    num_runs=100,
    occupancy=0.95,
    max_steps=500,
    snapshot_ticks=None,
    output_dir="GENERATED_DATA/rule_trees",
    verbose=True,
    skip_existing=True,
):
    """Run a tree num_runs times and save the runs to a single JSON file.

    If `skip_existing` is True (default) and the target JSON already
    exists, return immediately without regenerating. This makes the
    function safe to re-run after a partial failure: previously
    completed trees are left untouched.
    """
    if snapshot_ticks is None:
        snapshot_ticks = [100, 200, 300, 400, 500]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    filename = out_path / "ruletree_{:04d}.json".format(label)

    if skip_existing and filename.exists():
        if verbose:
            print("  tree label {}: SKIP (already exists)".format(label))
        return

    runs = []
    for i in range(num_runs):
        run = run_ruletree(
            grid_size=grid_size,
            tree=tree,
            occupancy=occupancy,
            max_steps=max_steps,
            snapshot_ticks=snapshot_ticks,
            seed=i,
        )
        run["label"] = label
        runs.append(run)

    with open(filename, "w") as f:
        json.dump(runs, f)
    if verbose:
        print("  tree label {}: {} runs -> {}".format(label, len(runs), filename))


def _generate_one_tree_worker(args):
    """Multiprocessing worker: generate all runs for one tree.

    Returns (label, skipped) so the caller can track progress.
    skipped=True means the output file already existed and nothing was
    regenerated. The output is written directly to disk by the worker
    (one JSON file per tree), avoiding any large IPC payloads.
    """
    (label, tree, kwargs) = args
    filename = Path(kwargs["output_dir"]) / "ruletree_{:04d}.json".format(label)
    skipped = kwargs.get("skip_existing", True) and filename.exists()
    generate_ruletree_data(tree, label, verbose=False, **kwargs)
    return label, skipped


def generate_all_ruletrees(
    tree_library,
    grid_size=50,
    num_runs=100,
    occupancy=0.95,
    max_steps=500,
    snapshot_ticks=None,
    output_dir="GENERATED_DATA/rule_trees",
    n_workers=1,
    skip_existing=True,
):
    """Generate simulation data for every tree in the library.

    tree_library: list of (label, Node, tree_str) tuples (as returned by
                  generate_tree_library or load_tree_library).
    n_workers:    number of multiprocessing workers. n_workers=1 uses the
                  serial path. With n_workers > 1, each worker generates
                  the full set of runs for one tree and writes its own
                  ruletree_NNNN.json file. Output is bit-identical to the
                  serial path because each run uses a deterministic seed.
    skip_existing: if True (default), trees whose ruletree_NNNN.json
                  already exists in output_dir are skipped. Makes the
                  function safe to re-run after a partial failure.
    """
    print("Generating rule-tree data for {} trees ({} workers, skip_existing={})...".format(
        len(tree_library), n_workers, skip_existing))

    common_kwargs = dict(
        grid_size=grid_size,
        num_runs=num_runs,
        occupancy=occupancy,
        max_steps=max_steps,
        snapshot_ticks=snapshot_ticks,
        output_dir=output_dir,
        skip_existing=skip_existing,
    )

    if n_workers <= 1:
        for label, tree, tree_str in tree_library:
            generate_ruletree_data(tree, label, **common_kwargs)
    else:
        from multiprocessing import Pool
        args_iter = [(label, tree, common_kwargs)
                     for label, tree, _ in tree_library]
        # Each task is heavy (full simulation x num_runs); use chunksize=1
        # so workers pick up new trees as they finish (better load balance
        # since trees vary in equilibrium time).
        completed = 0
        skipped = 0
        generated = 0
        with Pool(n_workers) as pool:
            for label, was_skipped in pool.imap_unordered(
                _generate_one_tree_worker, args_iter, chunksize=1
            ):
                completed += 1
                if was_skipped:
                    skipped += 1
                else:
                    generated += 1
                if completed % 10 == 0 or completed == len(tree_library):
                    print("  progress {}/{} ({} generated, {} skipped)".format(
                        completed, len(tree_library), generated, skipped))
    print("Done.")
