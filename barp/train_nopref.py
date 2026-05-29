"""Version A trainer: BaRP without preference encoding.

Implements Algorithm 1 lines 4-13 with z_t = h_t and r_t = q_t. The trainer
only sees the chosen action's reward because all access goes through env.observe.

Usage:
    python -m barp.train_nopref
    python -m barp.train_nopref --steps 10000 --batch-size 256 --lr 3e-4 --beta 0.01
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .env import RouterBenchBandit
from .model import BaRPNoPref
from .utils import pick_device


METRIC_COLS = [
    "step", "loss", "mean_reward", "mean_quality", "mean_cost_usd",
    "policy_entropy_nats", "val_quality", "val_cost_usd", "val_action_entropy_bits",
]


@torch.no_grad()
def evaluate(
    model: BaRPNoPref, env: RouterBenchBandit, split: str, device: torch.device,
    batch_size: int = 1024,
) -> dict[str, float]:
    """Algorithm 1 line 14: a* = argmax pi(a|x). Report mean quality and cost."""
    model.eval()
    idx_all, Q, C = env.qc_matrix(split)
    actions = np.zeros(len(idx_all), dtype=np.int64)
    for start in range(0, len(idx_all), batch_size):
        stop = min(start + batch_size, len(idx_all))
        h = torch.from_numpy(np.asarray(env.X[idx_all[start:stop]], dtype=np.float32)).to(device)
        actions[start:stop] = model(h).argmax(-1).cpu().numpy()
    model.train()
    rows = np.arange(len(idx_all))
    counts = np.bincount(actions, minlength=env.n_actions) / len(idx_all)
    return {
        "mean_quality": float(Q[rows, actions].mean()),
        "mean_cost_usd": float(C[rows, actions].mean()),
        "action_entropy_bits": float(-(counts * np.log2(counts.clip(min=1e-12))).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/nopref"))
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--beta", type=float, default=0.01, help="entropy bonus coeff")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"device: {device}")

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    env = RouterBenchBandit(args.data_dir)
    print(f"loaded bandit table: N={env.X.shape[0]:,}  A={env.n_actions}  embed_dim={env.embed_dim}")

    model = BaRPNoPref(env.embed_dim, env.n_actions, args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    run_dir = args.out_dir / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str))

    log_f = (run_dir / "metrics.csv").open("w", newline="")
    log_writer = csv.writer(log_f)
    log_writer.writerow(METRIC_COLS)

    best_val_q = -float("inf")
    best_ckpt = run_dir / "policy.pt"
    val_metrics: dict[str, float] = {}

    for step in range(1, args.steps + 1):
        idx, h_np = env.sample_batch(rng, split="train", batch_size=args.batch_size)
        h = torch.from_numpy(h_np).to(device)

        logits = model(h)                                       # line 5/6
        log_pi = logits.log_softmax(-1)
        pi = log_pi.exp()
        a = torch.distributions.Categorical(probs=pi).sample()  # line 7

        q_np, c_np = env.observe(a.cpu().numpy(), idx)          # line 8
        q = torch.from_numpy(q_np).to(device)
        c = torch.from_numpy(c_np).to(device)
        r = q                                                   # line 9 (Version A)
        b = r.mean().detach()                                   # line 10

        logp_a = log_pi.gather(1, a.unsqueeze(1)).squeeze(1)
        entropy = -(pi * log_pi).sum(-1)
        loss = -((r - b).detach() * logp_a).mean() - args.beta * entropy.mean()  # line 11

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()                                        # line 12

        if step % args.log_every == 0 or step == 1:
            print(
                f"step {step:6d}  loss={loss.item():+.4f}  r={r.mean().item():.4f}  "
                f"q={q.mean().item():.4f}  c=${c.mean().item():.4f}  H(pi)={entropy.mean().item():.3f}"
            )

        if step % args.val_every == 0 or step == args.steps:
            val_metrics = evaluate(model, env, "val", device)
            print(
                f"  [val] quality={val_metrics['mean_quality']:.4f}  "
                f"cost=${val_metrics['mean_cost_usd']:.4f}  "
                f"action_H={val_metrics['action_entropy_bits']:.2f} bits"
            )
            if val_metrics["mean_quality"] > best_val_q:
                best_val_q = val_metrics["mean_quality"]
                torch.save({
                    "state_dict": model.state_dict(),
                    "args": vars(args),
                    "models": env.models,
                    "step": step,
                    "val_quality": best_val_q,
                }, best_ckpt)

        log_writer.writerow([
            step, loss.item(), r.mean().item(), q.mean().item(), c.mean().item(),
            entropy.mean().item(),
            val_metrics.get("mean_quality", float("nan")),
            val_metrics.get("mean_cost_usd", float("nan")),
            val_metrics.get("action_entropy_bits", float("nan")),
        ])

    log_f.close()
    print(f"best val quality = {best_val_q:.4f}; checkpoint at {best_ckpt}")


if __name__ == "__main__":
    main()
