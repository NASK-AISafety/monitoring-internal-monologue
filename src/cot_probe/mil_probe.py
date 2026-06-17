import torch
import torch.nn as nn
from typing import List

activations = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


class PerLayerProbe(nn.Module):
    """Per-layer probe that transforms hidden states and optionally max-pools.

    Architecture:
        transform: Linear(d_model, hidden_dim) -> ReLU
        classifier: Linear(hidden_dim, 1)

    If input is 3D (batch, seq, d_model):
        transform -> max-pool over seq -> classifier
    If input is 2D (batch, d_model):
        transform -> classifier  (no pooling)
    """

    def __init__(self, d_model: int, hidden_dims: List[int], activation: str = "gelu"):
        super().__init__()
        layers = [nn.Linear(d_model, hidden_dims[0]), activations[activation]()]
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            layers.append(activations[activation]())
        self.transform = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dims[-1], 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cummax: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq, d_model) or (batch, d_model)
            attention_mask: (batch, seq) — 1 for valid tokens, 0 for padding.
                Only used when hidden_states is 3D.
            use_cummax: if True, use cumulative max instead of global max-pool.
                Returns per-position logits: (batch, seq, 1) instead of (batch, 1).

        Returns:
            logits: (batch, 1) or (batch, seq, 1) if use_cummax
        """
        needs_pooling = hidden_states.ndim == 3

        # Non-linear transform into latent space
        latent = self.transform(hidden_states)  # (..., hidden_dim)

        if needs_pooling:
            if attention_mask is not None:
                # Mask padding positions with -inf before max-pool
                mask = attention_mask.unsqueeze(-1).bool()  # (batch, seq, 1)
                latent = latent.masked_fill(~mask, float("-inf"))

            if use_cummax:
                # Cumulative max along sequence: (batch, seq, hidden_dim)
                latent, _ = torch.cummax(latent, dim=1)
            else:
                # Global max-pool over sequence dimension
                latent, _ = latent.max(dim=1)  # (batch, hidden_dim)

        logits = self.classifier(latent)  # (batch, 1) or (batch, seq, 1)
        return logits


class MILMultiLayerProbe(nn.Module):
    """Multi-layer MIL probe combining per-layer probes with a meta-layer.

    Each layer gets its own PerLayerProbe. Their outputs are combined
    via a learned meta-layer to produce the final prediction.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        hidden_dims: List[int],
        activation: str = "gelu",
    ):
        super().__init__()
        self.layer_probes = nn.ModuleList(
            [PerLayerProbe(d_model, hidden_dims, activation) for _ in range(num_layers)]
        )
        self.meta_layer = nn.Linear(num_layers, 1)

    def forward(
        self,
        hidden_states_per_layer: list[torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        use_cummax: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states_per_layer: list of tensors, each (batch, seq, d_model)
                or (batch, d_model). Length must equal num_layers.
            attention_mask: (batch, seq) — shared across layers.
            use_cummax: if True, return per-position predictions using cumulative max.

        Returns:
            final_logits: (batch, 1) or (batch, seq, 1) if use_cummax
            per_layer_logits: (batch, num_layers) or (batch, seq, num_layers)
        """
        per_layer_logits = []
        for layer_hidden_states, probe in zip(
            hidden_states_per_layer, self.layer_probes
        ):
            logits = probe(
                layer_hidden_states,
                attention_mask=attention_mask,
                use_cummax=use_cummax,
            )
            per_layer_logits.append(logits)

        # (batch, num_layers) or (batch, seq, num_layers) if use_cummax
        per_layer_logits = torch.cat(per_layer_logits, dim=-1)
        # (batch, 1) or (batch, seq, 1) if use_cummax
        final_logits = self.meta_layer(per_layer_logits)

        return final_logits, per_layer_logits
