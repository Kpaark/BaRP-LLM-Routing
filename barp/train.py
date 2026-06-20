"""Version B trainer: BaRP with MLP preference encoder and cost-aware reward.

Implements Algorithm 1 lines 4-13 in full:
    line 4: sample (x_t, w_t) -- w_t ~ Dirichlet(1, 1) per prompt
    line 5: u_t  = phi(w_t)
    line 6: z_t  = [h_t ; u_t]; pi_t = softmax(g_theta(z_t))
    line 7: a_t ~ pi_t
    line 8: observe q_t, c_t  (bandit feedback for the chosen action only)
    line 9: r_t = w_q * q_t - w_c * min(c_t / tau, 1)
    line 10-12: REINFORCE step with batch baseline + entropy bonus

`tau` is the cost cap that maps the raw USD cost to [0, 1]. If --tau is not
provided we read `tau_candidates_usd.p95` from data/meta.json (~$0.008 on the
default RouterBench cache).

Usage:
    python -m barp.train
    python -m barp.train --steps 10000 --batch-size 256 --lr 3e-4 --beta 0.01
    python -m barp.train --tau 0.01      # override the cost cap
    python -m barp.train --wandb --wandb-project barp-llm-routing --wandb-run-name my_run
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
from .model import BaRP
from .utils import pick_device
from .wandb_utils import finish as wandb_finish
from .wandb_utils import log_split_labels, log_train_step, log_val_step, maybe_init_wandb


METRIC_COLS = [
    "step", "loss", "mean_reward", "mean_quality", "mean_cost_usd",
    "mean_w_q", "mean_w_c", "policy_entropy_nats",
    "val_quality", "val_cost_usd", "val_action_entropy_bits",
]

# The fixed preference used for validation: a single, comparable point.
# w_q = w_c = 0.5 sits in the middle of the simplex.
VAL_W = (0.5, 0.5)


@torch.no_grad()
def evaluate(
    model: BaRP, env: RouterBenchBandit, split: str, device: torch.device,
    w: tuple[float, float] = VAL_W, batch_size: int = 1024,
) -> dict[str, float]:
    """Algorithm 1 line 14: a* = argmax pi(a|x, w). Report mean quality and cost
    at a fixed preference so validation curves stay comparable across steps."""
    model.eval()
    idx_all, Q, C = env.qc_matrix(split)
    w_t = torch.tensor([w], dtype=torch.float32, device=device)
    actions = np.zeros(len(idx_all), dtype=np.int64)
    for start in range(0, len(idx_all), batch_size):
        stop = min(start + batch_size, len(idx_all))
        h = torch.from_numpy(np.asarray(env.X[idx_all[start:stop]], dtype=np.float32)).to(device)
        w_b = w_t.expand(stop - start, -1)
        actions[start:stop] = model(h, w_b).argmax(-1).cpu().numpy()
    model.train()
    rows = np.arange(len(idx_all))
    counts = np.bincount(actions, minlength=env.n_actions) / len(idx_all)
    return {
        "mean_quality": float(Q[rows, actions].mean()),
        "mean_cost_usd": float(C[rows, actions].mean()),
        "action_entropy_bits": float(-(counts * np.log2(counts.clip(min=1e-12))).sum()),
    }


def resolve_tau(args_tau: float | None, data_dir: Path) -> float:
    """Use --tau if given; otherwise default to meta.json's p95 training cost."""
    if args_tau is not None:
        return float(args_tau)
    meta = json.loads((data_dir / "meta.json").read_text())
    return float(meta["tau_candidates_usd"]["p95"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/barp"))
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--beta", type=float, default=0.01, help="entropy bonus coeff")
    parser.add_argument("--tau", type=float, default=None,
                        help="cost cap in USD; defaults to meta.json's p95 (~$0.008)")
    parser.add_argument("--pref-hidden", type=int, default=256)
    parser.add_argument("--pref-out", type=int, default=768)
    parser.add_argument("--head-hidden", type=int, default=256)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wandb", action="store_true", help="log metrics to Weights & Biases")
    parser.add_argument("--wandb-project", default="barp-llm-routing")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-entity", default=None, help="W&B team/username (optional)")
    args = parser.parse_args()

    device = pick_device(args.device)
    tau = resolve_tau(args.tau, args.data_dir)
    print(f"device: {device}   tau=${tau:.5f}")

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    env = RouterBenchBandit(args.data_dir)
    print(f"loaded bandit table: N={env.X.shape[0]:,}  A={env.n_actions}  embed_dim={env.embed_dim}")

    model = BaRP(
        embed_dim=env.embed_dim,
        n_actions=env.n_actions,
        pref_dim=2,
        pref_hidden=args.pref_hidden,
        pref_out=args.pref_out,
        head_hidden=args.head_hidden,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    run_dir = args.out_dir / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = {**vars(args), "resolved_tau": tau}
    (run_dir / "args.json").write_text(
        json.dumps(run_config, indent=2, default=str)
    )

    wb = maybe_init_wandb(
        enabled=args.wandb,
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        entity=args.wandb_entity,
        config=run_config,
        run_dir=run_dir,
    )
    log_split_labels(wb, args.data_dir)

    log_f = (run_dir / "metrics.csv").open("w", newline="")
    log_writer = csv.writer(log_f)
    log_writer.writerow(METRIC_COLS)

    best_val_q = -float("inf")
    best_ckpt = run_dir / "policy.pt"
    val_metrics: dict[str, float] = {}

    for step in range(1, args.steps + 1):
        idx, h_np = env.sample_batch(rng, split="train", batch_size=args.batch_size)
        h = torch.from_numpy(h_np).to(device)
        # line 4: w_t ~ uniform on the 1-simplex. Dirichlet(1, 1) over 2 dims is
        # equivalent to drawing w_c ~ U(0, 1); using torch.rand keeps this op on
        # every backend (MPS lacks an aten::_sample_dirichlet kernel).
        w_c_sample = torch.rand(args.batch_size, device=device)
        w = torch.stack([1.0 - w_c_sample, w_c_sample], dim=-1)

        logits = model(h, w)                                    # lines 5-6: u = phi(w); z = [h;u]
        log_pi = logits.log_softmax(-1)
        pi = log_pi.exp()
        a = torch.distributions.Categorical(probs=pi).sample()  # line 7

        q_np, c_np = env.observe(a.cpu().numpy(), idx)          # line 8
        q = torch.from_numpy(q_np).to(device)
        c = torch.from_numpy(c_np).to(device)
        c_tilde = (c / tau).clamp(max=1.0)                      # cost normalized + capped
        r = w[:, 0] * q - w[:, 1] * c_tilde                     # line 9: cost-aware reward
        b = r.mean().detach()                                   # line 10: batch baseline

        logp_a = log_pi.gather(1, a.unsqueeze(1)).squeeze(1)
        entropy = -(pi * log_pi).sum(-1)
        loss = -((r - b).detach() * logp_a).mean() - args.beta * entropy.mean()  # line 11

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()                                        # line 12

        if step % args.log_every == 0 or step == 1:
            print(
                f"step {step:6d}  loss={loss.item():+.4f}  r={r.mean().item():+.4f}  "
                f"q={q.mean().item():.4f}  c=${c.mean().item():.4f}  "
                f"w_q={w[:, 0].mean().item():.2f}  H(pi)={entropy.mean().item():.3f}"
            )
            log_train_step(
                wb, step,
                loss=loss.item(),
                reward=r.mean().item(),
                quality=q.mean().item(),
                cost_usd=c.mean().item(),
                w_q=w[:, 0].mean().item(),
                w_c=w[:, 1].mean().item(),
                policy_entropy=entropy.mean().item(),
            )

        if step % args.val_every == 0 or step == args.steps:
            val_metrics = evaluate(model, env, "val", device)
            print(
                f"  [val @ w=(0.5, 0.5)] quality={val_metrics['mean_quality']:.4f}  "
                f"cost=${val_metrics['mean_cost_usd']:.4f}  "
                f"action_H={val_metrics['action_entropy_bits']:.2f} bits"
            )
            log_val_step(
                wb, step,
                quality=val_metrics["mean_quality"],
                cost_usd=val_metrics["mean_cost_usd"],
                action_entropy_bits=val_metrics["action_entropy_bits"],
            )
            if val_metrics["mean_quality"] > best_val_q:
                best_val_q = val_metrics["mean_quality"]
                torch.save({
                    "state_dict": model.state_dict(),
                    "args": vars(args),
                    "resolved_tau": tau,
                    "models": env.models,
                    "step": step,
                    "val_quality": best_val_q,
                }, best_ckpt)

        log_writer.writerow([
            step, loss.item(), r.mean().item(), q.mean().item(), c.mean().item(),
            w[:, 0].mean().item(), w[:, 1].mean().item(),
            entropy.mean().item(),
            val_metrics.get("mean_quality", float("nan")),
            val_metrics.get("mean_cost_usd", float("nan")),
            val_metrics.get("action_entropy_bits", float("nan")),
        ])

    log_f.close()
    wandb_finish(wb)
    print(f"best val quality @ w=(0.5, 0.5) = {best_val_q:.4f}; checkpoint at {best_ckpt}")


if __name__ == "__main__":
    main()
