"""Axial patch Transformer over continuous fish density/momentum fields.

This is the field-ablation counterpart of genmod/models/traj_transformer.py.
It consumes the [B, T, G, G, C] fields produced by genmod/fish/field.py, where
each frame is a coarse render of one kick event. The design deliberately
mirrors two existing models so the ablation differs only where the observation
forces it to:

    from traj_transformer.py   axial attention, CLS-over-time pooling, embed()
    from transformer.py        decomposed row/col spatial position embeddings
                               (as in TransformerRuleClassifierRowCol)

nn.Embedding cannot be reused for the input stage: fields are continuous, so
the patch stage is a linear projection of each 2x2xC patch, exactly as the
Schelling row/col model projects discrete patches through an embedding.

AXIAL, not flat. A single frame is G/patch squared = 256 patch tokens at
G=32; over T=64 frames that is 16384 tokens, and a flat attention over all of
them is O(16384^2) and infeasible. Axial attention factorizes it: a SPATIAL
attention over the 256 patches within each frame, then a TEMPORAL attention
over the T frames within each patch position (weights shared across
positions). Cost is O(T * 256^2 + 256 * T^2), the same trick traj_transformer
uses to keep long windows affordable.

The irregular inter-kick interval dt is injected as an additive per-frame
scalar embedding, so the model conditions on the same timing signal the
trajectory model sees as a feature column.
"""

import torch
import torch.nn as nn


def _encoder_layer(d_model, n_heads, dropout):
    return nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=4 * d_model,
        dropout=dropout,
        batch_first=True,
        activation="gelu",
        norm_first=True,
    )


class FieldTransformer(nn.Module):
    """Classify the generating rule from a [B, T, G, G, C] field window."""

    def __init__(
        self,
        in_channels: int = 3,
        patch: int = 2,
        grid: int = 32,
        max_events: int = 64,
        num_classes: int = 11,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if grid % patch != 0:
            raise ValueError("grid %d not divisible by patch %d" % (grid, patch))
        self.patch = patch
        self.grid = grid
        self.gp = grid // patch                 # patches per side
        self.n_patches = self.gp * self.gp
        self.in_channels = in_channels
        self.d_model = d_model
        self.num_classes = num_classes

        # Continuous patch projection: flatten each 2x2xC patch and project.
        self.proj = nn.Linear(patch * patch * in_channels, d_model)

        # Decomposed position: time (per frame) + row + col (per patch), the
        # same row/col decomposition as the Schelling row/col ablation.
        self.time_emb = nn.Embedding(max_events, d_model)
        self.row_emb = nn.Embedding(self.gp, d_model)
        self.col_emb = nn.Embedding(self.gp, d_model)
        # Per-frame irregular interval, injected as an additive scalar embedding.
        self.dt_proj = nn.Linear(1, d_model)

        # One axial block = spatial attention (over patches), then temporal
        # attention (over frames). Same structure as traj_transformer.
        self.spatial = nn.ModuleList(
            [_encoder_layer(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.temporal = nn.ModuleList(
            [_encoder_layer(d_model, n_heads, dropout) for _ in range(n_layers)])

        # Pool over patches by mean (keeps the model spatially permutation
        # tolerant given the position tags), then a CLS token pools over time.
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)
        self.pool = _encoder_layer(d_model, n_heads, dropout)

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        row_ids = torch.arange(self.n_patches) // self.gp
        col_ids = torch.arange(self.n_patches) % self.gp
        self.register_buffer("row_ids", row_ids, persistent=False)
        self.register_buffer("col_ids", col_ids, persistent=False)

    def _patchify(self, field):
        """[B, T, G, G, C] -> [B, T, N, patch*patch*C]."""
        B, T, G, _, C = field.shape
        p, gp = self.patch, self.gp
        x = field.view(B, T, gp, p, gp, p, C)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()   # [B,T,gp,gp,p,p,C]
        return x.view(B, T, gp * gp, p * p * C)

    def _trunk(self, field, dt):
        """[B, T, G, G, C] field and [B, T] dt -> pooled [B, d]."""
        B, T = field.shape[0], field.shape[1]
        d = self.d_model
        N = self.n_patches

        h = self.proj(self._patchify(field))              # [B, T, N, d]
        h = h + self.row_emb(self.row_ids)[None, None]
        h = h + self.col_emb(self.col_ids)[None, None]
        t_idx = torch.arange(T, device=field.device)
        h = h + self.time_emb(t_idx)[None, :, None, :]
        h = h + self.dt_proj(dt.unsqueeze(-1))[:, :, None, :]

        for spatial, temporal in zip(self.spatial, self.temporal):
            # Spatial: one sequence per frame, over the N patch positions.
            hs = h.reshape(B * T, N, d)
            hs = spatial(hs)
            h = hs.reshape(B, T, N, d)
            # Temporal: one sequence per patch position, over the T frames.
            ht = h.permute(0, 2, 1, 3).reshape(B * N, T, d)
            ht = temporal(ht)
            h = ht.reshape(B, N, T, d).permute(0, 2, 1, 3)

        h = h.mean(dim=2)                                  # over patches: [B,T,d]
        h = torch.cat([self.cls.expand(B, -1, -1), h], dim=1)
        return self.ln_f(self.pool(h)[:, 0])

    def forward(self, field, dt):
        return self.head(self._trunk(field, dt))

    @torch.no_grad()
    def embed(self, field, dt):
        """Pooled representation, for the t-SNE and linear-probe analyses."""
        return self._trunk(field, dt)
