"""Offline bandit environment built from the cached RouterBench arrays.

    env = RouterBenchBandit("data/")
    idx, h = env.sample_batch(rng, split="train", batch_size=256)   # Alg. 1 line 4
    q, c   = env.observe(actions, idx)                              # Alg. 1 line 8

`observe` reveals q_t and c_t only for the chosen action -- the bandit constraint.
`qc_matrix` returns the full Q/C arrays for a split and is for eval/oracle only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class RouterBenchBandit:
    def __init__(self, data_dir: str | Path = "data") -> None:
        data_dir = Path(data_dir)
        self.X = np.load(data_dir / "X.npy", mmap_mode="r")
        self.Q = np.load(data_dir / "Q.npy", mmap_mode="r")
        self.C = np.load(data_dir / "C.npy", mmap_mode="r")
        self.splits: dict[str, np.ndarray] = {
            k: np.asarray(v, dtype=np.int64)
            for k, v in json.loads((data_dir / "splits.json").read_text()).items()
        }
        self.models: list[str] = json.loads((data_dir / "models.json").read_text())
        self.n_actions = len(self.models)
        self.embed_dim = int(self.X.shape[1])

    def sample_batch(
        self, rng: np.random.Generator, split: str, batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        idx = rng.choice(self.splits[split], size=batch_size, replace=True)
        return idx, np.asarray(self.X[idx], dtype=np.float32)

    def observe(
        self, actions: np.ndarray, indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bandit feedback: q_t and c_t for the chosen action a_t only."""
        actions = np.asarray(actions, dtype=np.int64)
        indices = np.asarray(indices, dtype=np.int64)
        return (
            np.asarray(self.Q[indices, actions], dtype=np.float32),
            np.asarray(self.C[indices, actions], dtype=np.float32),
        )

    def qc_matrix(self, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Full (q, c) matrices for a split. Eval / oracle only -- never training."""
        idx = self.splits[split]
        return (
            idx,
            np.asarray(self.Q[idx], dtype=np.float32),
            np.asarray(self.C[idx], dtype=np.float32),
        )
