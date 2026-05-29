"""Evaluate a saved BaRPNoPref policy against four reference baselines.

Baselines on the test split:
    random       -- a ~ Uniform(0, A)
    cheapest     -- argmin_a mean cost on train
    best-single  -- argmax_a mean quality on train
    oracle       -- per-prompt argmax_a quality (upper bound)

Usage:
    python -m barp.eval --checkpoint runs/nopref/<ts>/policy.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .env import RouterBenchBandit
from .model import BaRPNoPref
from .utils import pick_device


@torch.no_grad()
def policy_actions(
    model: BaRPNoPref, env: RouterBenchBandit, indices: np.ndarray,
    device: torch.device, batch_size: int = 1024,
) -> np.ndarray:
    """a* = argmax_a pi(a|x) over the given prompt indices."""
    model.eval()
    out = np.zeros(len(indices), dtype=np.int64)
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        h = torch.from_numpy(np.asarray(env.X[indices[start:stop]], dtype=np.float32)).to(device)
        out[start:stop] = model(h).argmax(-1).cpu().numpy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = pick_device(args.device)
    env = RouterBenchBandit(args.data_dir)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    hidden_dim = ckpt["args"].get("hidden_dim", 256)

    model = BaRPNoPref(env.embed_dim, env.n_actions, hidden_dim).to(device)
    model.load_state_dict(ckpt["state_dict"])

    eval_idx, Q_eval, C_eval = env.qc_matrix(args.split)
    rows = np.arange(len(eval_idx))

    _, Q_train, C_train = env.qc_matrix("train")
    cheapest = int(np.argmin(C_train.mean(0)))
    best_single = int(np.argmax(Q_train.mean(0)))

    rng = np.random.default_rng(0)
    routers = [
        ("random", rng.integers(0, env.n_actions, size=len(eval_idx))),
        (f"cheapest ({env.models[cheapest]})", np.full(len(eval_idx), cheapest, dtype=np.int64)),
        (f"best-single ({env.models[best_single]})", np.full(len(eval_idx), best_single, dtype=np.int64)),
        ("BaRPNoPref (this)", policy_actions(model, env, eval_idx, device)),
        ("oracle (per-prompt argmax q)", Q_eval.argmax(-1)),
    ]

    def summary(name: str, a: np.ndarray) -> dict:
        counts = np.bincount(a, minlength=env.n_actions)
        return {
            "router": name,
            "mean_quality": float(Q_eval[rows, a].mean()),
            "mean_cost_usd": float(C_eval[rows, a].mean()),
            "top_action": env.models[int(counts.argmax())],
            "action_distribution_pct": dict(zip(env.models, (counts / len(a) * 100).round(1).tolist())),
        }

    results = [summary(name, a) for name, a in routers]

    print(f"\nEvaluation on split = {args.split}  (N = {len(eval_idx):,})")
    print(f"{'router':40s}  {'quality':>8s}  {'cost ($)':>10s}  top action")
    print("-" * 80)
    for r in results:
        print(f"{r['router']:40s}  {r['mean_quality']:8.4f}  {r['mean_cost_usd']:10.5f}  {r['top_action']}")

    out_path = args.checkpoint.parent / f"eval_{args.split}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
