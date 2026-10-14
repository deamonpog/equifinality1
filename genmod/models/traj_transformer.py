"""Transformer encoder over continuous agent-time tokens.

The Schelling model in genmod/models/transformer.py embeds DISCRETE patch
tokens with nn.Embedding(vocab_size, d_model). Trajectories are continuous,
so the input stage is replaced by a shared MLP projection; the encoder, the
CLS pooling, and the temperature-scaling wrapper mirror that model so the two
case studies differ only where the domain forces them to.

AXIAL attention over a [T, 5] grid of agent-time states. Each block applies
two attentions in sequence:

    temporal   attention over the T events WITHIN each agent, with weights
               shared across agents. This is what lets the model follow an
               individual agent along its own trajectory.

    social     attention over the 5 agents WITHIN each event. A pure set
               operation, so it cannot depend on agent numbering.

Why not one flat attention over all 5T tokens.

The obvious design -- concatenate the 5T agent-time cells into one sequence
with a time embedding and no agent embedding -- is exactly permutation
invariant, and it is WRONG. With no agent id, that model sees an unordered
BAG of (state, time) tokens: nothing links "agent 3 at event 10" to "agent 3
at event 11". It is therefore also invariant to shuffling the agents
INDEPENDENTLY at each event, which is not a symmetry at all -- it dices every
trajectory into disconnected fragments.

That destroys exactly the information this task turns on. Whether an agent
KEEPS the same nearest neighbor from kick to kick separates "nearest-k" from
"random-k", and nearest-neighbor persistence was among the most
discriminative statistics in the univariate screen. Identity across time must
survive; identity across agents must not matter.

Axial attention delivers both, and is also cheaper: O(A*T^2 + T*A^2) instead
of O((A*T)^2), about a 5x saving at A=5, which is what makes long windows
affordable.

Permutation invariance BY CONSTRUCTION. There is still no agent-id embedding,
the temporal weights are shared across agents, the social attention is a set
operation, and the final pooling averages over agents. So

    f(W) = f(PW)   for any global relabelling P, and for ANY weights

Being architectural rather than learned, this holds at initialization and
after training alike; scripts/fish/verify_permutation_invariance.py asserts
it, together with the control that a per-event shuffle DOES change the output.
Random agent permutation is still applied during training as a cheap
redundancy against an implementation slip.
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


class TrajectoryTransformer(nn.Module):
    """Classify the generating rule from a [B, T, 5, F] trajectory window."""

    def __init__(
        self,
        n_features: int,
        n_agents: int,
        max_events: int,
        num_classes: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.num_classes = num_classes
        self.d_model = d_model

        # Shared across agents: the same projection is applied to every
        # (agent, event) cell. A per-agent projection would break permutation
        # invariance immediately.
        self.proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Time embedding only. No agent embedding, by design.
        self.time_emb = nn.Embedding(max_events, d_model)

        # One axial block = temporal attention, then social attention.
        self.temporal = nn.ModuleList(
            [_encoder_layer(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.social = nn.ModuleList(
            [_encoder_layer(d_model, n_heads, dropout) for _ in range(n_layers)])

        # Pooling. Averaging over agents first is what makes the whole model
        # permutation invariant; the CLS token then pools over time.
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)
        self.pool = _encoder_layer(d_model, n_heads, dropout)

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def _trunk(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, A, F] -> pooled [B, d]."""
        B, T, A, _ = x.shape
        d = self.d_model

        h = self.proj(x)                                   # [B, T, A, d]
        t_idx = torch.arange(T, device=x.device)
        h = h + self.time_emb(t_idx)[None, :, None, :]

        for temporal, social in zip(self.temporal, self.social):
            # Time axis, one sequence per agent. Weights are shared across
            # agents, so this preserves each agent's identity along its own
            # trajectory without making the model care which agent it is.
            h = h.permute(0, 2, 1, 3).reshape(B * A, T, d)
            h = temporal(h)
            h = h.reshape(B, A, T, d).permute(0, 2, 1, 3)

            # Agent axis, one sequence per event. A set operation over the
            # five agents: no ordering information exists here at all.
            h = h.reshape(B * T, A, d)
            h = social(h)
            h = h.reshape(B, T, A, d)

        h = h.mean(dim=2)                                  # over agents: [B,T,d]
        h = torch.cat([self.cls.expand(B, -1, -1), h], dim=1)
        return self.ln_f(self.pool(h)[:, 0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self._trunk(x))

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled representation, for the t-SNE and linear-probe analyses."""
        return self._trunk(x)


class GroupGRU(nn.Module):
    """Recurrent baseline over per-event GROUP statistics.

    The control that separates two explanations of any Transformer win. This
    model is also a sequence model, and also sees the full time course, but it
    consumes a single group-level vector per event: the agents have already
    been aggregated away. If the Transformer beats it, the gain comes from
    preserving individual-agent structure, not merely from modelling the
    sequence (RQ5, ablation 2).
    """

    def __init__(self, n_features: int, num_classes: int,
                 hidden: int = 128, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, num_layers=n_layers,
                          batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
                          bidirectional=False)
        self.ln_f = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, G] per-event group statistics. Returns logits."""
        out, _ = self.gru(x)
        return self.head(self.ln_f(out[:, -1]))


class TemperatureScaled(nn.Module):
    """Wrap a trained model with a single learned temperature.

    Temperature is fitted on the VALIDATION split, never on calibration: the
    calibration split has one job, which is to set the conformal quantile, and
    reusing it to tune the temperature would make the RAPS coverage guarantee
    conditional on data it has already seen.
    """

    def __init__(self, model: nn.Module, temperature: float = 1.0):
        super().__init__()
        self.model = model
        self.log_t = nn.Parameter(torch.tensor(float(temperature)).log())

    @property
    def temperature(self) -> float:
        return float(self.log_t.exp())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x) / self.log_t.exp()
