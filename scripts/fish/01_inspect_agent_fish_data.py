"""Inspect the simulated-agent schooling-fish dataset (Lei et al. 2020 release).

This script makes no changes to the data. It verifies every structural
assumption the rest of the pipeline depends on, prints a human-readable
report, and writes results/fish/inspection.json for downstream scripts.

Claims verified here:
  1. Each file has 5 columns (ExperimentID, AgentID, Time, X, Y), no header.
  2. Every run contains exactly 5 agents.
  3. Within a run, all 5 agents share an identical time vector, so a run is
     a clean [T, 5, 2] tensor on a shared (irregular) grid.
  4. Experiment IDs are seed-linked across rule files, which forces the
     grouped-split design.
  5. The 11 rule classes are balanced.
  6. The arena is centered at the origin with radius R = 0.25 m.

Run from the repository root:
    python scripts/fish/01_inspect_agent_fish_data.py
"""

import json
import os
import re
from collections import Counter

import numpy as np

DATA_DIR = os.path.join("FISH_DATA", "Agents")
OUT_PATH = os.path.join("results", "fish", "inspection.json")

# Rule table from the paper: (strategy family, number of selected neighbors).
# The rule id is the index into this list.
RULE_TABLE = [
    ("none", 0),
    ("nearest", 1),
    ("nearest", 2),
    ("nearest", 3),
    ("random", 1),
    ("random", 2),
    ("random", 3),
    ("mostinfluential", 1),
    ("mostinfluential", 2),
    ("mostinfluential", 3),
    ("all", 4),
]

FAMILY_FROM_FILENAME = {
    "no-interaction": "none",
    "nearest": "nearest",
    "random": "random",
    "mostinfluential": "mostinfluential",
    "all-neighbors": "all",
}

ARENA_RADIUS = 0.25  # metres, from the paper (agents in a 25 cm radius tank)


def parse_filename(name):
    """Map '2.agent-random-k3.dat' -> (rule_id, family, k).

    The leading digit is the strategy family index used by the authors; we
    ignore it and recover the family from the descriptive part of the name,
    then look the (family, k) pair up in RULE_TABLE to get our rule id.
    """
    m = re.match(r"^\d+\.agent-(.+)-k(\d+)\.dat$", name)
    if m is None:
        raise ValueError("unexpected filename: %s" % name)
    family_token, k = m.group(1), int(m.group(2))
    if family_token not in FAMILY_FROM_FILENAME:
        raise ValueError("unknown strategy family in %s" % name)
    family = FAMILY_FROM_FILENAME[family_token]
    rule_id = RULE_TABLE.index((family, k))
    return rule_id, family, k


def describe(values):
    """Summary statistics for a 1-D array, as plain Python floats."""
    v = np.asarray(values, dtype=float)
    return {
        "min": float(v.min()),
        "p01": float(np.percentile(v, 1)),
        "median": float(np.median(v)),
        "mean": float(v.mean()),
        "p99": float(np.percentile(v, 99)),
        "max": float(v.max()),
    }


def inspect_file(path):
    """Load one rule file and check every structural claim about it."""
    raw = np.loadtxt(path)

    report = {}
    report["n_rows"] = int(raw.shape[0])
    report["n_cols"] = int(raw.shape[1])
    report["n_nan"] = int(np.isnan(raw).sum())

    # Duplicate rows: an exact repeat would mean a logging error.
    n_unique = len(np.unique(raw, axis=0))
    report["n_duplicate_rows"] = int(raw.shape[0] - n_unique)

    exp = raw[:, 0].astype(int)
    agent = raw[:, 1].astype(int)
    time = raw[:, 2]
    xy = raw[:, 3:5]

    exp_ids = np.unique(exp)
    report["n_experiments"] = int(exp_ids.size)
    report["experiment_id_range"] = [int(exp_ids.min()), int(exp_ids.max())]
    report["agent_ids"] = [int(a) for a in np.unique(agent)]

    # --- per-run checks -------------------------------------------------
    events_per_run = []
    dt_all = []
    runs_with_five_agents = 0
    runs_with_shared_time = 0
    run_durations = []
    first_events = {}  # exp_id -> [5, 3] array of (t, x, y) at the first event

    for e in exp_ids:
        m = exp == e
        agents_here = np.unique(agent[m])
        if agents_here.size == 5:
            runs_with_five_agents += 1

        # Time vector per agent, in the order the rows appear.
        per_agent_time = [time[m & (agent == a)] for a in agents_here]
        lengths = {t.size for t in per_agent_time}
        shared = len(lengths) == 1 and all(
            np.allclose(t, per_agent_time[0]) for t in per_agent_time
        )
        if shared:
            runs_with_shared_time += 1

        n_events = per_agent_time[0].size
        events_per_run.append(n_events)
        run_durations.append(float(per_agent_time[0][-1] - per_agent_time[0][0]))
        dt_all.append(np.diff(per_agent_time[0]))

        first = np.stack(
            [
                np.concatenate([[time[m & (agent == a)][0]], xy[m & (agent == a)][0]])
                for a in agents_here
            ]
        )
        first_events[int(e)] = first

    report["runs_with_five_agents"] = runs_with_five_agents
    report["runs_with_shared_time_vector"] = runs_with_shared_time
    report["events_per_run"] = describe(events_per_run)
    report["run_duration_s"] = describe(run_durations)
    report["dt_s"] = describe(np.concatenate(dt_all))
    report["n_nonpositive_dt"] = int((np.concatenate(dt_all) <= 0).sum())

    # --- arena geometry -------------------------------------------------
    radius = np.sqrt((xy ** 2).sum(axis=1))
    report["radius_m"] = describe(radius)
    report["n_outside_arena"] = int((radius > ARENA_RADIUS).sum())
    report["centroid_xy"] = [float(xy[:, 0].mean()), float(xy[:, 1].mean())]

    return report, first_events


def main():
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".dat"))
    print("Found %d agent files in %s\n" % (len(files), DATA_DIR))

    # Show the raw bytes of one file so we can see the format with our own eyes.
    with open(os.path.join(DATA_DIR, files[0]), "r") as fh:
        head = [next(fh).rstrip("\n") for _ in range(3)]
    print("First 3 raw lines of %s:" % files[0])
    for line in head:
        print("   |%s|" % line)
    print("   columns: ExperimentID  AgentID  Time(s)  X(m)  Y(m)\n")

    reports = {}
    first_events_by_rule = {}

    print("%-34s %8s %5s %6s %7s %7s %6s" % (
        "file", "rows", "exps", "5agts", "shared", "ev/run", "dupes"))
    print("-" * 82)

    for name in files:
        rule_id, family, k = parse_filename(name)
        rep, first_events = inspect_file(os.path.join(DATA_DIR, name))
        rep["file"] = name
        rep["rule_id"] = rule_id
        rep["strategy_family"] = family
        rep["neighbor_count"] = k
        reports[name] = rep
        first_events_by_rule[name] = first_events

        print("%-34s %8d %5d %6d %7d %7.0f %6d" % (
            name,
            rep["n_rows"],
            rep["n_experiments"],
            rep["runs_with_five_agents"],
            rep["runs_with_shared_time_vector"],
            rep["events_per_run"]["median"],
            rep["n_duplicate_rows"],
        ))

    # ---- claim 1: structure ------------------------------------------
    print("\n[1] Structure")
    all_5cols = all(r["n_cols"] == 5 for r in reports.values())
    all_5agents = all(
        r["runs_with_five_agents"] == r["n_experiments"] for r in reports.values())
    all_shared = all(
        r["runs_with_shared_time_vector"] == r["n_experiments"]
        for r in reports.values())
    no_nan = all(r["n_nan"] == 0 for r in reports.values())
    no_dupes = all(r["n_duplicate_rows"] == 0 for r in reports.values())
    print("    every file has 5 columns .................. %s" % all_5cols)
    print("    every run has exactly 5 agents ............ %s" % all_5agents)
    print("    every run has a shared time vector ........ %s" % all_shared)
    print("    no missing values ........................ %s" % no_nan)
    print("    no duplicate rows ........................ %s" % no_dupes)
    if all_shared:
        print("    => each run is a clean [T, 5, 2] tensor.")
    else:
        print("    => WARNING: some runs need interpolation. Stop and rethink.")

    # ---- claim 2: timing ---------------------------------------------
    dts = [r["dt_s"] for r in reports.values()]
    med_dt = float(np.mean([d["median"] for d in dts]))
    ev = [r["events_per_run"]["median"] for r in reports.values()]
    print("\n[2] Timing (event-indexed axis)")
    print("    median inter-kick interval dt ............ %.3f s" % med_dt)
    print("    dt range ................................. %.3f to %.3f s" % (
        min(d["min"] for d in dts), max(d["max"] for d in dts)))
    print("    non-positive dt (must be 0) .............. %d" % sum(
        r["n_nonpositive_dt"] for r in reports.values()))
    print("    median events per run .................... %.0f" % np.median(ev))
    for T in (16, 32, 64, 128):
        print("      T=%3d kicks  ~= %6.1f s of observation, %4.0f windows/run "
              "at stride T" % (T, T * med_dt, np.median(ev) / T))

    # ---- claim 3: arena ----------------------------------------------
    print("\n[3] Arena geometry")
    rmax = max(r["radius_m"]["max"] for r in reports.values())
    cx = float(np.mean([r["centroid_xy"][0] for r in reports.values()]))
    cy = float(np.mean([r["centroid_xy"][1] for r in reports.values()]))
    print("    max observed radius ...................... %.4f m" % rmax)
    print("    assumed arena radius R ................... %.4f m" % ARENA_RADIUS)
    print("    positions outside R ...................... %d" % sum(
        r["n_outside_arena"] for r in reports.values()))
    print("    mean position (should be near 0, 0) ...... (%.4f, %.4f)" % (cx, cy))

    # ---- claim 4: seed sharing across rules ---------------------------
    # If experiment e starts from the same seed family in every rule file,
    # its first-event state will coincide (fully or partially) across files.
    # Any such coupling forces experiment id to be the grouping key.
    print("\n[4] Seed sharing across rule files (drives the split design)")
    names = list(first_events_by_rule)
    exp_ids = sorted(first_events_by_rule[names[0]])
    n_identical = 0
    n_partial = 0
    for e in exp_ids:
        states = [first_events_by_rule[n][e] for n in names]
        ref = states[0]
        if all(np.allclose(s, ref, atol=1e-6) for s in states):
            n_identical += 1
        else:
            # Partial coupling: any single (agent, coordinate) cell that is
            # identical across all 11 rule files is evidence of a shared seed.
            same_cells = np.ones(ref.shape, dtype=bool)
            for s in states[1:]:
                same_cells &= np.isclose(s, ref, atol=1e-6)
            if same_cells.any():
                n_partial += 1
    print("    experiments with an IDENTICAL first event in all 11 files . %d/%d"
          % (n_identical, len(exp_ids)))
    print("    experiments with a PARTIALLY shared first event ........... %d/%d"
          % (n_partial, len(exp_ids)))
    if n_identical + n_partial > 0:
        print("    => experiment ids ARE seed-linked across rules.")
        print("       Experiment e must go to the same partition in all 11")
        print("       rule files, or the classifier can memorize the initial")
        print("       condition instead of learning the behavioral rule.")
    else:
        print("    => no detectable coupling; grouping by run would suffice.")
        print("       (We will still group by experiment id: it is strictly safer.)")

    # ---- claim 5: class balance ---------------------------------------
    print("\n[5] Class balance")
    counts = Counter()
    for r in reports.values():
        counts[r["rule_id"]] = r["n_experiments"]
    total = sum(counts.values())
    print("    rule  strategy          k   runs")
    for rid in range(len(RULE_TABLE)):
        fam, k = RULE_TABLE[rid]
        print("    %4d  %-16s %2d   %4d" % (rid, fam, k, counts[rid]))
    print("    total runs ............................... %d" % total)
    balanced = len(set(counts.values())) == 1
    print("    balanced ................................. %s" % balanced)
    print("    chance top-1 accuracy .................... %.4f" % (1.0 / len(RULE_TABLE)))

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    summary = {
        "arena_radius_m": ARENA_RADIUS,
        "n_rules": len(RULE_TABLE),
        "rule_table": [{"rule_id": i, "strategy_family": f, "neighbor_count": k}
                       for i, (f, k) in enumerate(RULE_TABLE)],
        "median_dt_s": med_dt,
        "seed_sharing": {"identical": n_identical, "partial": n_partial,
                         "n_experiments": len(exp_ids)},
        "class_balanced": balanced,
        "per_file": reports,
    }
    with open(OUT_PATH, "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\nWrote %s" % OUT_PATH)


if __name__ == "__main__":
    main()
