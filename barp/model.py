"""Policy networks for BaRP.

Version A: `BaRPNoPref` -- prompt embedding only, no preference encoder.
Version B: `BaRP`       -- adds an MLP phi(w) and concatenates u = phi(w) to
                           the prompt embedding before the decision head.
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


class BaRP(nn.Module):
    """Algorithm 1 with preference encoding: z_t = [h_t ; phi(w_t)].

    phi:  (B, pref_dim) -> (B, pref_out)        -- preference encoder MLP
    head: (B, embed_dim + pref_out) -> (B, n_actions) logits

    forward(h, w): joint embedding -> logits over LLMs.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        n_actions: int = 11,
        pref_dim: int = 2,
        pref_hidden: int = 64,
        pref_out: int = 32,
        head_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(pref_dim, pref_hidden),
            nn.ReLU(),
            nn.Linear(pref_hidden, pref_out),
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim + pref_out, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, n_actions),
        )

    def forward(self, h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        u = self.phi(w)
        z = torch.cat([h, u], dim=-1)
        return self.head(z)
