"""Sanity check: does the trained BaRP policy actually condition on `w`?

Loads a Version B checkpoint and evaluates the argmax policy at six fixed
preference points on the simplex (the two extremes plus the four interior
points the paper sweeps). For each `w_c` we report mean quality, mean cost,
the top action, and the full action distribution.

Expected behavior:
    w_c = 0.0  (quality-only)  -> top action = the highest-quality LLM (GPT-4)
    w_c = 1.0  (cost-only)     -> top action = the cheapest LLM (Mistral-7B)
    intermediate w_c           -> a graded mix; top action shifts as w_c grows

Failure modes:
    * Same top action at every w_c     -> preference encoder is being ignored.
    * Top action does not shift cheaper as w_c grows -> reward / model bug.
    * action distribution collapses to a single LLM at every w_c -> entropy
      collapsed too early; retrain with --beta 0.05.

Usage:
    python -m barp.eval_extremes --checkpoint runs/barp/<ts>/policy.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .env import RouterBenchBandit
from .model import BaRP
from .utils import pick_device


# Six points on the simplex: the two extremes plus the paper's four interior
# samples. Each entry is w_c (so w_q = 1 - w_c).
DEFAULT_W_C = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


@torch.no_grad()
def policy_actions(
    model: BaRP, env: RouterBenchBandit, indices: np.ndarray, w_c: float,
    device: torch.device, batch_size: int = 1024,
) -> np.ndarray:
    """a* = argmax_a pi(a | x, w) at a fixed preference."""
    model.eval()
    w_row = torch.tensor([[1.0 - w_c, w_c]], dtype=torch.float32, device=device)
    out = np.zeros(len(indices), dtype=np.int64)
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        h = torch.from_numpy(np.asarray(env.X[indices[start:stop]], dtype=np.float32)).to(device)
        w = w_row.expand(stop - start, -1)
        out[start:stop] = model(h, w).argmax(-1).cpu().numpy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--w-c", type=float, nargs="+", default=list(DEFAULT_W_C),
                        help="cost weights to sweep; w_q = 1 - w_c")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = pick_device(args.device)
    env = RouterBenchBandit(args.data_dir)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    ckpt_args = ckpt["args"]
    model = BaRP(
        embed_dim=env.embed_dim,
        n_actions=env.n_actions,
        pref_dim=2,
        pref_hidden=ckpt_args.get("pref_hidden", 256),
        pref_out=ckpt_args.get("pref_out", 768),
        head_hidden=ckpt_args.get("head_hidden", 256),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])

    eval_idx, Q_eval, C_eval = env.qc_matrix(args.split)
    rows = np.arange(len(eval_idx))

    results: list[dict] = []
    for w_c in args.w_c:
        a = policy_actions(model, env, eval_idx, w_c, device)
        counts = np.bincount(a, minlength=env.n_actions)
        results.append({
            "w_c": float(w_c),
            "w_q": float(1.0 - w_c),
            "mean_quality": float(Q_eval[rows, a].mean()),
            "mean_cost_usd": float(C_eval[rows, a].mean()),
            "top_action": env.models[int(counts.argmax())],
            "action_distribution_pct": dict(
                zip(env.models, (counts / len(a) * 100).round(1).tolist())
            ),
        })

    print(f"\nExtremes sweep on split = {args.split}  (N = {len(eval_idx):,})")
    print(f"checkpoint: {args.checkpoint}")
    print(f"resolved tau (training): ${ckpt.get('resolved_tau', float('nan')):.5f}")
    print(f"\n{'w_c':>5s}  {'w_q':>5s}  {'quality':>8s}  {'cost ($)':>10s}  top action")
    print("-" * 70)
    for r in results:
        print(
            f"{r['w_c']:5.2f}  {r['w_q']:5.2f}  "
            f"{r['mean_quality']:8.4f}  {r['mean_cost_usd']:10.5f}  {r['top_action']}"
        )

    out_path = args.checkpoint.parent / "eval_extremes.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
