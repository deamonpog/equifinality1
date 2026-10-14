"""Load the raw simulated-agent .dat files into per-run position tensors.

The raw format is whitespace-delimited with no header and five columns:

    ExperimentID  AgentID  Time(s)  X(m)  Y(m)

Rows are grouped by agent, and within a run all five agents share an
identical time vector (verified for all 550 runs by
scripts/fish/01_inspect_agent_fish_data.py). A run therefore reduces to a
time vector of length T and a [T, 5, 2] position tensor.
"""

import csv
import os
from dataclasses import dataclass
from typing import List

import numpy as np

N_AGENTS = 5
ARENA_RADIUS = 0.25  # metres


@dataclass
class Run:
    """One simulation run: five agents under one known rule."""

    rule_id: int
    family_id: int
    neighbor_count: int
    exp_id: int
    time: np.ndarray  # [T] kick times, seconds, strictly increasing
    pos: np.ndarray   # [T, 5, 2] positions, metres

    @property
    def run_id(self) -> str:
        return "r%02d_e%02d" % (self.rule_id, self.exp_id)

    @property
    def n_events(self) -> int:
        return int(self.time.shape[0])


def load_rule_lookup(path=os.path.join("data", "fish", "rule_lookup.csv")):
    """Read the canonical label table written by 02_create_rule_lookup.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            "%s not found. Run scripts/fish/02_create_rule_lookup.py first." % path)
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for key in ("rule_id", "family_id", "neighbor_count", "n_runs"):
            r[key] = int(r[key])
    return rows


def load_runs(data_dir=os.path.join("FISH_DATA", "Agents"),
              lookup_path=os.path.join("data", "fish", "rule_lookup.csv")) -> List[Run]:
    """Load every run of every rule into a flat list of Run objects."""
    lookup = load_rule_lookup(lookup_path)
    runs = []

    for entry in lookup:
        raw = np.loadtxt(os.path.join(data_dir, entry["source_file"]))
        exp = raw[:, 0].astype(int)
        agent = raw[:, 1].astype(int)

        for e in np.unique(exp):
            m = exp == e
            t_ref = None
            pos = np.empty((0, N_AGENTS, 2))

            for a in range(1, N_AGENTS + 1):
                sel = m & (agent == a)
                t_a = raw[sel, 2]
                xy_a = raw[sel, 3:5]
                if t_ref is None:
                    t_ref = t_a
                    pos = np.empty((t_ref.shape[0], N_AGENTS, 2), dtype=np.float64)
                elif t_a.shape != t_ref.shape or not np.allclose(t_a, t_ref):
                    # Step 01 proved this cannot happen; fail loudly if it ever does.
                    raise ValueError(
                        "run %s exp %d agent %d has a divergent time vector"
                        % (entry["source_file"], e, a))
                pos[:, a - 1, :] = xy_a

            runs.append(Run(
                rule_id=entry["rule_id"],
                family_id=entry["family_id"],
                neighbor_count=entry["neighbor_count"],
                exp_id=int(e),
                time=t_ref,
                pos=pos,
            ))

    return runs
