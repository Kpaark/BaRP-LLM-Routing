"""Policy networks for BaRP.

Version A (this file): BaRPNoPref - prompt embedding only, no preference encoder.
Version B (future):    BaRP       - adds an MLP phi(w) and concatenates u = phi(w)
                                    to the prompt embedding before the head.
"""

from __future__ import annotations

import torch
from torch import nn


class BaRPNoPref(nn.Module):
    """Algorithm 1 with the preference branch removed: z_t = h_t.

    forward(h): (B, embed_dim) -> (B, n_actions) logits; softmax gives pi_t.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        n_actions: int = 11,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h)
