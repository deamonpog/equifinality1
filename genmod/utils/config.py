"""Configuration loading from YAML with defaults.

Single system: rule_trees (Schelling segregation with auto-generated rule trees).
"""

from dataclasses import dataclass, field
from typing import List

import yaml


@dataclass
class DataConfig:
    system: str = "rule_trees"
    data_dir: str = "GENERATED_DATA"

    # Common splits
    train_frac: float = 0.8
    val_frac: float = 0.1

    # Schelling grid (also used by rule_trees)
    grid_size: int = 50
    schelling_patch_size: int = 2
    num_snapshots: int = 5

    # Rule tree generation
    n_tree_candidates: int = 10000
    tree_max_depth: int = 3


@dataclass
class ModelConfig:
    model_size: str = "base"
    dropout: float = 0.1
    positional_encoding: str = "time_space"  # "time_space" or "standard"


@dataclass
class TrainingConfig:
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    epochs: int = 30
    label_smoothing: float = 0.05
    seed: int = 42
    device: str = "cuda"
    lr_scheduler: str = "cosine"
    warmup_epochs: int = 3
    checkpoint_dir: str = "results/checkpoints"
    save_every: int = 5
    log_dir: str = "results/logs"


@dataclass
class ConformalConfig:
    enabled: bool = True
    method: str = "raps"
    alpha_levels: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.10, 0.20])
    k_reg: int = 5
    lambda_reg: float = 0.01
    cal_fraction: float = 0.5  # fraction of val set used for conformal calibration


@dataclass
class ExperimentConfig:
    name: str = "default"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    conformal: ConformalConfig = field(default_factory=ConformalConfig)


def load_config(path: str) -> ExperimentConfig:
    """Load experiment config from YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    cfg = ExperimentConfig()
    cfg.name = raw.get("name", "default")

    for section_name, section_obj in [("data", cfg.data), ("model", cfg.model),
                                       ("training", cfg.training), ("conformal", cfg.conformal)]:
        if section_name in raw:
            for k, v in raw[section_name].items():
                if hasattr(section_obj, k):
                    setattr(section_obj, k, v)

    return cfg


def save_config(cfg: ExperimentConfig, path: str) -> None:
    """Save experiment config to YAML file."""
    from dataclasses import asdict
    with open(path, "w") as f:
        yaml.dump(asdict(cfg), f, default_flow_style=False)
