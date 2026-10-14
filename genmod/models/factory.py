"""Model factory: build models from config with size presets."""

from typing import Dict

from genmod.models.transformer import (
    TransformerRuleClassifier,
    TransformerRuleClassifierRowCol,
    TransformerRuleClassifierStdPos,
)


MODEL_SIZE_PRESETS: Dict[str, Dict[str, int]] = {
    "tiny":   {"d_model": 64,  "n_heads": 2, "n_layers": 2},
    "small":  {"d_model": 128, "n_heads": 4, "n_layers": 2},
    "medium": {"d_model": 128, "n_heads": 4, "n_layers": 4},
    "base":   {"d_model": 256, "n_heads": 8, "n_layers": 4},
    "large":  {"d_model": 384, "n_heads": 8, "n_layers": 6},
}


def build_model(
    num_classes: int,
    vocab_size: int,
    seq_len: int,
    time_size: int,
    space_size: int,
    model_size: str = "base",
    dropout: float = 0.1,
    positional_encoding: str = "time_space",
    cls_token_id: int = None,
    grid_rows: int = None,
    grid_cols: int = None,
) -> TransformerRuleClassifier:
    """Build a Transformer classifier from explicit tokenization params.

    System-agnostic: works for any system that provides vocab_size, seq_len,
    time_size, space_size. The cls_token_id is accepted for interface
    compatibility but not used by the model itself (the dataset embeds CLS
    into the token sequence).

    positional_encoding:
        - "time_space": decomposed time + flat space embeddings (default)
        - "time_row_col": decomposed time + separate row/col embeddings
          (requires grid_rows and grid_cols)
        - "standard": single learned sequence-position embedding
    """
    if model_size not in MODEL_SIZE_PRESETS:
        raise ValueError(f"Unknown model_size: {model_size}. Choose from {list(MODEL_SIZE_PRESETS)}")

    preset = MODEL_SIZE_PRESETS[model_size]

    if positional_encoding == "time_row_col":
        if grid_rows is None or grid_cols is None:
            raise ValueError("positional_encoding='time_row_col' requires grid_rows and grid_cols")
        return TransformerRuleClassifierRowCol(
            vocab_size=vocab_size,
            seq_len=seq_len,
            num_classes=num_classes,
            d_model=preset["d_model"],
            n_heads=preset["n_heads"],
            n_layers=preset["n_layers"],
            dropout=dropout,
            time_size=time_size,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
        )

    model_cls = TransformerRuleClassifier if positional_encoding == "time_space" else TransformerRuleClassifierStdPos

    return model_cls(
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_classes=num_classes,
        d_model=preset["d_model"],
        n_heads=preset["n_heads"],
        n_layers=preset["n_layers"],
        dropout=dropout,
        time_size=time_size,
        space_size=space_size,
    )


def get_tokenization_params(system: str, **kwargs) -> dict:
    """Dispatch tokenization params by system name."""
    if system == "rule_trees":
        from genmod.data.ruletree_dataset import get_ruletree_tokenization_params
        return get_ruletree_tokenization_params(**kwargs)
    elif system == "schelling":
        from genmod.data.schelling_dataset import get_schelling_tokenization_params
        return get_schelling_tokenization_params(**kwargs)
    else:
        raise ValueError(f"Unknown system: {system}")
