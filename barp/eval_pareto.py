"""Sweep `w_c` at fine granularity and plot the cost-vs-quality Pareto curve.

Loads a Version B checkpoint, runs the argmax policy at `--n-points` evenly
spaced preference values along the simplex, and overlays:

    * the BaRP Pareto curve (one point per `w_c`, colored by w_c),
    * the 11 single-model points (each LLM as the always-pick choice),
    * the oracle (per-prompt argmax q -- unrealistic upper bound),
    * a random baseline (uniform over models).

Outputs:
    figures/pareto.png                       -- the headline figure
    runs/barp/<ts>/pareto.json               -- raw points (ckpt-relative)

Usage:
    python -m barp.eval_pareto --checkpoint runs/barp/<ts>/policy.pt
    python -m barp.eval_pareto --checkpoint runs/barp/<ts>/policy.pt --n-points 41
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm

from .env import RouterBenchBandit
from .model import BaRP
from .utils import pick_device


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


def short_name(model_name: str) -> str:
    """Strip the vendor prefix for legend / annotation labels."""
    return model_name.split("/", 1)[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--n-points", type=int, default=21,
                        help="number of w_c values along [0, 1]")
    parser.add_argument("--figure-path", type=Path, default=Path("figures/pareto.png"))
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

    # 1) BaRP Pareto curve: argmax inference at each w_c
    w_cs = np.linspace(0.0, 1.0, args.n_points)
    barp_points: list[dict] = []
    for w_c in w_cs:
        a = policy_actions(model, env, eval_idx, float(w_c), device)
        barp_points.append({
            "w_c": float(w_c),
            "mean_quality": float(Q_eval[rows, a].mean()),
            "mean_cost_usd": float(C_eval[rows, a].mean()),
        })

    # 2) Single-model reference points
    single_points = [
        {
            "model": m,
            "mean_quality": float(Q_eval[:, j].mean()),
            "mean_cost_usd": float(C_eval[:, j].mean()),
        }
        for j, m in enumerate(env.models)
    ]

    # 3) Oracle (per-prompt argmax q) and random baselines
    oracle_actions = Q_eval.argmax(-1)
    oracle_point = {
        "mean_quality": float(Q_eval[rows, oracle_actions].mean()),
        "mean_cost_usd": float(C_eval[rows, oracle_actions].mean()),
    }
    rng = np.random.default_rng(0)
    random_actions = rng.integers(0, env.n_actions, size=len(eval_idx))
    random_point = {
        "mean_quality": float(Q_eval[rows, random_actions].mean()),
        "mean_cost_usd": float(C_eval[rows, random_actions].mean()),
    }

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(9, 6))

    barp_q = np.array([p["mean_quality"] for p in barp_points])
    barp_c = np.array([p["mean_cost_usd"] for p in barp_points])
    sc = ax.scatter(barp_c, barp_q, c=w_cs, cmap="viridis", s=40, zorder=4,
                    edgecolors="black", linewidths=0.5, label="BaRP (varying w_c)")
    ax.plot(barp_c, barp_q, color="gray", alpha=0.4, zorder=3)

    for p in single_points:
        ax.scatter(p["mean_cost_usd"], p["mean_quality"], marker="s",
                   color="#d62728", s=70, zorder=5, edgecolors="black", linewidths=0.5)
        ax.annotate(short_name(p["model"]),
                    (p["mean_cost_usd"], p["mean_quality"]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=8, color="#7a1d1d")

    ax.scatter(oracle_point["mean_cost_usd"], oracle_point["mean_quality"],
               marker="*", color="gold", s=300, zorder=6,
               edgecolors="black", linewidths=0.8, label="Oracle (per-prompt argmax q)")
    ax.scatter(random_point["mean_cost_usd"], random_point["mean_quality"],
               marker="^", color="gray", s=80, zorder=5,
               edgecolors="black", linewidths=0.5, label="Random")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("w_c (cost weight)")

    ax.set_xlabel("mean cost per prompt (USD)")
    ax.set_ylabel("mean quality")
    ax.set_title(f"BaRP cost-quality Pareto curve  (split = {args.split},  N = {len(eval_idx):,})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    args.figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.figure_path, dpi=150)
    plt.close(fig)

    # ---- Save raw points next to the checkpoint ----
    out_path = args.checkpoint.parent / "pareto.json"
    out_path.write_text(json.dumps({
        "split": args.split,
        "n_eval": int(len(eval_idx)),
        "checkpoint": str(args.checkpoint),
        "resolved_tau": float(ckpt.get("resolved_tau", float("nan"))),
        "barp_curve": barp_points,
        "single_models": single_points,
        "oracle": oracle_point,
        "random": random_point,
    }, indent=2))

    print(f"BaRP curve points: {len(barp_points)}")
    print(f"  w_c=0.0:  q={barp_points[0]['mean_quality']:.4f}  c=${barp_points[0]['mean_cost_usd']:.5f}")
    print(f"  w_c=1.0:  q={barp_points[-1]['mean_quality']:.4f}  c=${barp_points[-1]['mean_cost_usd']:.5f}")
    print(f"Oracle  :  q={oracle_point['mean_quality']:.4f}  c=${oracle_point['mean_cost_usd']:.5f}")
    print(f"Random  :  q={random_point['mean_quality']:.4f}  c=${random_point['mean_cost_usd']:.5f}")
    print(f"\nWrote {args.figure_path}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
