"""Generate a library of rule trees and their Schelling simulation data.

This is the data-generation entry point for Option D. It supports three
modes so the same script can drive a single-job pipeline OR a chained
multi-stage pipeline (e.g. an LSF or Slurm job array on a cluster):

  Default (no mode flag): generate the library AND simulate every tree.

  --library_only:         generate the library, save tree_library.json,
                          then exit. Use this for stage 1 of a chained
                          pipeline.

  --simulate_only:        load an existing tree_library.json, skip
                          generation/dedup, then simulate either the
                          full library or a slice (see --task_id).

  --simulate_only --n_tasks N --task_id I:
                          simulate only the slice
                          library[I*total//N : (I+1)*total//N].
                          With N tasks running in parallel, every label
                          in [0, total) is covered exactly once. Used
                          for stage 2 of an array-based pipeline.

This is a heavy job and is intended to run on HPC, not locally.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genmod.data.rule_trees import save_tree_library, load_tree_library
from genmod.data.tree_generators import generate_tree_library
from genmod.data.schelling_ruletree import generate_all_ruletrees


def parse_int_list(s):
    """Parse a comma-separated list of integers like '100,200,300'."""
    return [int(x) for x in s.split(",")]


def compute_slice(total, n_tasks, task_id):
    """Return (start, end) indices for task_id in an n_tasks split of [0,total).

    Uses floor division so the union of all slices is exactly [0, total)
    with no overlap and no gaps.
    """
    if n_tasks <= 0:
        raise ValueError("n_tasks must be >= 1, got {}".format(n_tasks))
    if not (0 <= task_id < n_tasks):
        raise ValueError("task_id must be in [0, n_tasks); got task_id={} n_tasks={}".format(
            task_id, n_tasks))
    start = task_id * total // n_tasks
    end = (task_id + 1) * total // n_tasks
    return start, end


def main():
    parser = argparse.ArgumentParser(description="Generate rule-tree dataset")

    # Mode flags (mutually compatible only as documented in the module docstring)
    parser.add_argument("--library_only", action="store_true",
                        help="Build the tree library and exit. Skip simulation.")
    parser.add_argument("--simulate_only", action="store_true",
                        help="Load an existing tree_library.json and only run "
                             "the simulation step. Skip generation and dedup.")
    parser.add_argument("--n_tasks", type=int, default=None,
                        help="When set together with --task_id and --simulate_only, "
                             "simulate only the slice of trees assigned to this task. "
                             "Total trees are split into n_tasks contiguous chunks.")
    parser.add_argument("--task_id", type=int, default=None,
                        help="0-based task index when using --n_tasks. "
                             "Each chained job array task should set this from "
                             "(LSB_JOBINDEX - 1) or SLURM_ARRAY_TASK_ID.")

    # Tree generation params (used unless --simulate_only)
    parser.add_argument("--n_candidates", type=int, default=10000,
                        help="Number of candidate trees to generate before dedup")
    parser.add_argument("--max_depth", type=int, default=3,
                        help="Maximum tree depth")
    parser.add_argument("--method", type=str, default="mixed",
                        choices=["pseudo", "quasi", "mixed"],
                        help="Tree generation method")

    # Simulation params (used unless --library_only)
    parser.add_argument("--num_runs", type=int, default=100,
                        help="Schelling simulation runs per tree")
    parser.add_argument("--grid_size", type=int, default=50)
    parser.add_argument("--occupancy", type=float, default=0.95)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--snapshot_ticks", type=str, default="100,200,300,400,500",
                        help="Comma-separated list of ticks at which to snapshot")

    # Common
    parser.add_argument("--output_dir", type=str, default="GENERATED_DATA")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of multiprocessing workers for fingerprinting "
                             "and simulation. 1 = serial. Set to the number of "
                             "available CPU cores on HPC.")
    args = parser.parse_args()

    if args.library_only and args.simulate_only:
        parser.error("--library_only and --simulate_only are mutually exclusive")

    if (args.n_tasks is None) != (args.task_id is None):
        parser.error("--n_tasks and --task_id must be used together")

    if args.n_tasks is not None and not args.simulate_only:
        parser.error("--n_tasks / --task_id only valid with --simulate_only")

    snapshot_ticks = parse_int_list(args.snapshot_ticks)
    out_root = Path(args.output_dir) / "rule_trees"
    library_path = out_root / "tree_library.json"

    print("=" * 60)
    print("Rule-tree data generation")
    print("=" * 60)
    if args.library_only:
        mode = "library_only"
    elif args.simulate_only:
        if args.n_tasks is not None:
            mode = "simulate_only (slice {}/{})".format(args.task_id, args.n_tasks)
        else:
            mode = "simulate_only (full library)"
    else:
        mode = "build + simulate"
    print("  mode        : {}".format(mode))
    print("  candidates  : {}".format(args.n_candidates))
    print("  max_depth   : {}".format(args.max_depth))
    print("  method      : {}".format(args.method))
    print("  num_runs    : {}".format(args.num_runs))
    print("  grid_size   : {}".format(args.grid_size))
    print("  max_steps   : {}".format(args.max_steps))
    print("  snapshots   : {}".format(snapshot_ticks))
    print("  output_dir  : {}".format(out_root))
    print("  seed        : {}".format(args.seed))
    print("  workers     : {}".format(args.workers))
    print()

    # ---- Build the library (default mode and library_only) ----
    if not args.simulate_only:
        library = generate_tree_library(
            n_candidates=args.n_candidates,
            max_depth=args.max_depth,
            seed=args.seed,
            method=args.method,
            n_workers=args.workers,
        )
        save_tree_library(library, library_path)
    else:
        if not library_path.exists():
            raise FileNotFoundError(
                "--simulate_only requires an existing tree library at {}. "
                "Run with --library_only first.".format(library_path))
        library = load_tree_library(library_path)
        print("Loaded existing tree library: {} trees".format(len(library)))

    if args.library_only:
        print()
        print("Library built: {} trees -> {}".format(len(library), library_path))
        return

    # ---- Slice the library if running as part of a job array ----
    if args.n_tasks is not None:
        total = len(library)
        start, end = compute_slice(total, args.n_tasks, args.task_id)
        sliced = library[start:end]
        print("Slice {}/{}: trees [{}, {}) of {} -> {} trees".format(
            args.task_id, args.n_tasks, start, end, total, len(sliced)))
        if not sliced:
            print("Empty slice; nothing to simulate.")
            return
        target_library = sliced
    else:
        target_library = library

    # ---- Simulate ----
    generate_all_ruletrees(
        target_library,
        grid_size=args.grid_size,
        num_runs=args.num_runs,
        occupancy=args.occupancy,
        max_steps=args.max_steps,
        snapshot_ticks=snapshot_ticks,
        output_dir=str(out_root),
        n_workers=args.workers,
    )

    print()
    print("Data generation complete: {} trees in this run, {} runs each".format(
        len(target_library), args.num_runs))


if __name__ == "__main__":
    main()
