"""Loading helper for rule-tree Schelling data.

The grid format is identical to the Schelling threshold experiments
(0/1/2 cells on an NxN grid), so we reuse SchellingDataset and
get_schelling_tokenization_params directly. This module only adds the
file-loading logic for ruletree_NNNN.json files.
"""

import json
from pathlib import Path
from typing import Dict, List

from genmod.data.schelling_dataset import get_schelling_tokenization_params


def load_ruletree_data(data_dir, num_classes) -> Dict[int, List[dict]]:
    """Load every ruletree_NNNN.json file in data_dir.

    Returns dict mapping label -> list of run dicts.
    """
    data_path = Path(data_dir)
    all_runs: Dict[int, List[dict]] = {}
    for label in range(num_classes):
        p = data_path / "ruletree_{:04d}.json".format(label)
        if not p.exists():
            raise FileNotFoundError(
                "Missing {}. Generate data first.".format(p))
        all_runs[label] = json.loads(p.read_text())
    return all_runs


def get_ruletree_tokenization_params(grid_size=50, patch_size=2, num_snapshots=5):
    """Tokenization parameters for rule-tree grids (delegates to Schelling)."""
    return get_schelling_tokenization_params(
        grid_size=grid_size,
        patch_size=patch_size,
        num_snapshots=num_snapshots,
    )
