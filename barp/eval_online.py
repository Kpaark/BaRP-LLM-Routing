"""Online adaptation on top of a frozen BaRP checkpoint: eps_t-greedy bandit.

Setup (n_arms = n_models + 1):
    * arms 0 .. n_models-1  -> always route to that fixed model
    * arm  n_models         -> ask frozen BaRP (at preference w_c) which model
                               to use for this prompt

The test split is treated as an online stream (shuffled once per seed). At
step t we choose an arm with eps_t-greedy:

    with prob eps_t      explore: uniform-random arm among all n_arms
    with prob 1 - eps_t  exploit: the GREEDY arm

    eps_t = min(eps_max, c / t)          (decaying exploration, ~c*ln(T)
                                          exploration pulls over T steps)

Greedy arm (--greedy):
    barp     the BaRP arm is *always* the greedy arm (requested scheme --
             exploit means "trust the pretrained router")
    argmax   classic eps_t-greedy control: exploit the arm with the highest
             empirical mean so far (optimistic init so every arm gets tried)

Feedback is bandit-style: we observe RouterBench quality only for the model
actually queried, and update that arm's empirical mean quality. The argmax
variant *uses* the means to act; the barp variant tracks them as an online
diagnostic ("is some fixed model beating BaRP on this stream?").

Usage:
    # Hard OOD (GSM8K stream, BaRP never trained on it):
    python -m barp.eval_online --data-dir data_ood_hard \\
        --checkpoint runs/ood_hard_gsm8k/<ts>/policy.pt

    # Classic argmax control on the same stream:
    python -m barp.eval_online --data-dir data_ood_hard \\
        --checkpoint runs/ood_hard_gsm8k/<ts>/policy.pt --greedy argmax
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .env import RouterBenchBandit
from .eval_table import policy_actions
from .model import BaRP
from .utils import pick_device
from .wandb_utils import finish as wandb_finish
from .wandb_utils import maybe_resume_wandb


def epsilon_schedule(t: int, c: float, eps_max: float) -> float:
    """eps_t = min(eps_max, c / t) for 1-indexed step t."""
    return min(eps_max, c / t)


def run_stream(
    *,
    Q_eval: np.ndarray,
    C_eval: np.ndarray,
    barp_actions: np.ndarray,
    greedy: str,
    eps_c: float,
    eps_max: float,
    seed: int,
    steps: int | None,
) -> dict:
    """One online pass over the (shuffled) test stream. Returns per-run stats.

    Q_eval/C_eval are (N, n_models); barp_actions is BaRP's per-prompt model
    choice (precomputed once -- the checkpoint is frozen, only the bandit
    layer on top adapts).
    """
    n_prompts, n_models = Q_eval.shape
    n_arms = n_models + 1
    barp_arm = n_models
    rng = np.random.default_rng(seed)

    T = steps or n_prompts
    # Stream order: reshuffled permutations, tiled if steps > n_prompts.
    reps = int(np.ceil(T / n_prompts))
    order = np.concatenate([rng.permutation(n_prompts) for _ in range(reps)])[:T]

    # Empirical means. Optimistic init (quality upper bound = 1.0) so the
    # argmax variant tries every arm before settling; for the barp variant
    # the init never influences action selection.
    counts = np.zeros(n_arms, dtype=np.int64)
    means = np.ones(n_arms, dtype=np.float64)

    qual = np.zeros(T, dtype=np.float64)
    cost = np.zeros(T, dtype=np.float64)
    arms = np.zeros(T, dtype=np.int64)
    eps_trace = np.zeros(T, dtype=np.float64)

    for t in range(1, T + 1):
        i = order[t - 1]
        eps_t = epsilon_schedule(t, eps_c, eps_max)
        eps_trace[t - 1] = eps_t

        if rng.random() < eps_t:
            arm = int(rng.integers(n_arms))              # explore
        elif greedy == "barp":
            arm = barp_arm                               # exploit = trust BaRP
        else:
            arm = int(np.argmax(means))                  # exploit = argmax

        model_a = int(barp_actions[i]) if arm == barp_arm else arm
        q = float(Q_eval[i, model_a])                    # bandit feedback
        c = float(C_eval[i, model_a])

        counts[arm] += 1
        means[arm] += (q - means[arm]) / counts[arm]     # incremental mean

        qual[t - 1] = q
        cost[t - 1] = c
        arms[t - 1] = arm

    running_avg = np.cumsum(qual) / np.arange(1, T + 1)
    return {
        "counts": counts,
        "means": means,
        "running_avg": running_avg,
        "avg_quality": float(qual.mean()),
        "avg_cost": float(cost.mean()),
        "eps_trace": eps_trace,
        "arms": arms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data_ood_hard"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--w-c", type=float, default=0.0,
                        help="preference fed to the frozen BaRP arm (default 0 = quality-focused)")
    parser.add_argument("--greedy", choices=["barp", "argmax"], default="barp",
                        help="greedy arm: 'barp' (requested scheme) or classic 'argmax' control")
    parser.add_argument("--eps-c", type=float, default=None,
                        help="c in eps_t = min(eps_max, c/t); default = n_arms")
    parser.add_argument("--eps-max", type=float, default=1.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="stream shuffles + coin flips; results averaged")
    parser.add_argument("--steps", type=int, default=None,
                        help="stream length T (default: one pass over the test split)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="barp-llm-routing")
    parser.add_argument("--wandb-run-id", default=None)
    parser.add_argument("--wandb-entity", default=None)
    args = parser.parse_args()

    device = pick_device(args.device)
    env = RouterBenchBandit(args.data_dir)

    meta_path = args.data_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    spec_name = (meta.get("split_spec") or {}).get("name", args.data_dir.name)

    eval_idx, Q_eval, C_eval = env.qc_matrix("test")
    n = len(eval_idx)
    n_models = env.n_actions
    n_arms = n_models + 1
    eps_c = args.eps_c if args.eps_c is not None else float(n_arms)

    # ----- frozen BaRP arm: precompute its per-prompt model choice -----
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
    barp_actions = policy_actions(model, env, eval_idx, args.w_c, device)

    # Positional actions within the stream (Q_eval rows are 0..n-1).
    # barp_actions is aligned with eval_idx order already.

    # ----- online runs across seeds -----
    runs = [
        run_stream(
            Q_eval=Q_eval, C_eval=C_eval, barp_actions=barp_actions,
            greedy=args.greedy, eps_c=eps_c, eps_max=args.eps_max,
            seed=s, steps=args.steps,
        )
        for s in args.seeds
    ]
    T = len(runs[0]["running_avg"])
    curves = np.stack([r["running_avg"] for r in runs])          # (S, T)
    curve_mean, curve_std = curves.mean(0), curves.std(0)
    final_avg = np.array([r["avg_quality"] for r in runs])
    final_cost = np.array([r["avg_cost"] for r in runs])
    counts_mean = np.stack([r["counts"] for r in runs]).mean(0)
    means_mean = np.stack([r["means"] for r in runs]).mean(0)

    # ----- reference policies on the same test set -----
    rows_ref = np.arange(n)
    pure_barp = float(Q_eval[rows_ref, barp_actions].mean())
    oracle = float(Q_eval.max(-1).mean())
    fixed_means = Q_eval.mean(0)                                  # hindsight per-model truth
    best_fixed_a = int(np.argmax(fixed_means))
    best_fixed = float(fixed_means[best_fixed_a])
    gpt4_a = env.models.index("gpt-4-1106-preview") if "gpt-4-1106-preview" in env.models else None
    gpt4 = float(fixed_means[gpt4_a]) if gpt4_a is not None else float("nan")

    arm_names = [*env.models, f"BaRP (w_c={args.w_c:.2f})"]

    # ----- report -----
    print(f"\nOnline eps_t-greedy on '{spec_name}' test stream  "
          f"(N = {n:,}, T = {T:,}, seeds = {args.seeds})")
    print(f"greedy arm: {args.greedy}   eps_t = min({args.eps_max:g}, {eps_c:g}/t)   "
          f"n_arms = {n_arms} ({n_models} models + BaRP)")
    print(f"checkpoint: {args.checkpoint}\n")

    print(f"{'Policy':<44s}{'Avg quality':>14s}{'Avg cost ($)':>14s}")
    print("-" * 72)
    print(f"{'Online ' + args.greedy + '-greedy (mean over seeds)':<44s}"
          f"{100 * final_avg.mean():>13.2f}±{100 * final_avg.std():.2f}"
          f"{final_cost.mean():>13.5f}")
    print(f"{'Pure BaRP (eps=0 limit)':<44s}{100 * pure_barp:>14.2f}"
          f"{float(C_eval[rows_ref, barp_actions].mean()):>14.5f}")
    print(f"{'Best fixed model (hindsight): ' + env.models[best_fixed_a]:<44s}"
          f"{100 * best_fixed:>14.2f}{float(C_eval[:, best_fixed_a].mean()):>14.5f}")
    if gpt4_a is not None:
        print(f"{'Always GPT-4':<44s}{100 * gpt4:>14.2f}"
              f"{float(C_eval[:, gpt4_a].mean()):>14.5f}")
    print(f"{'Oracle (per-prompt max)':<44s}{100 * oracle:>14.2f}{'':>14s}")

    print(f"\nPer-arm empirical means after the stream (avg over seeds):")
    print(f"{'Arm':<44s}{'pulls':>10s}{'emp. mean':>12s}{'true mean':>12s}")
    print("-" * 78)
    order_by_mean = np.argsort(-means_mean)
    for a in order_by_mean:
        true_m = 100 * pure_barp if a == n_models else 100 * float(fixed_means[a])
        pulled = counts_mean[a] > 0
        emp = f"{100 * means_mean[a]:.2f}" if pulled else "--"
        print(f"{arm_names[a]:<44s}{counts_mean[a]:>10.1f}{emp:>12s}{true_m:>12.2f}")

    # ----- save json + md -----
    stem = f"eval_online_greedy_{args.greedy}"
    out = {
        "spec_name": spec_name,
        "n_test": n,
        "T": T,
        "seeds": args.seeds,
        "greedy": args.greedy,
        "w_c": args.w_c,
        "eps_c": eps_c,
        "eps_max": args.eps_max,
        "n_arms": n_arms,
        "arm_names": arm_names,
        "online_avg_quality_mean": float(100 * final_avg.mean()),
        "online_avg_quality_std": float(100 * final_avg.std()),
        "online_avg_cost": float(final_cost.mean()),
        "pure_barp": 100 * pure_barp,
        "best_fixed_model": env.models[best_fixed_a],
        "best_fixed": 100 * best_fixed,
        "always_gpt4": 100 * gpt4,
        "oracle": 100 * oracle,
        "arm_pulls_mean": counts_mean.tolist(),
        "arm_empirical_means": (100 * means_mean).tolist(),
        "arm_true_means": (100 * fixed_means).tolist() + [100 * pure_barp],
        "running_avg_mean": (100 * curve_mean).tolist(),
        "running_avg_std": (100 * curve_std).tolist(),
    }
    out_json = args.checkpoint.parent / f"{stem}.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_json}")

    md = [
        f"# Online eps_t-greedy ({args.greedy}-greedy) on `{spec_name}`",
        "",
        f"- **stream:** test split, N = {n:,}, T = {T:,}, seeds = {args.seeds}",
        f"- **eps_t:** min({args.eps_max:g}, {eps_c:g}/t)   |   **arms:** {n_models} models + frozen BaRP (w_c={args.w_c:.2f})",
        f"- **checkpoint:** `{args.checkpoint}`",
        "",
        "| Policy | Avg quality | Avg cost ($) |",
        "| --- | --- | --- |",
        f"| Online {args.greedy}-greedy | {100 * final_avg.mean():.2f} ± {100 * final_avg.std():.2f} | {final_cost.mean():.5f} |",
        f"| Pure BaRP | {100 * pure_barp:.2f} | {float(C_eval[rows_ref, barp_actions].mean()):.5f} |",
        f"| Best fixed ({env.models[best_fixed_a]}) | {100 * best_fixed:.2f} | {float(C_eval[:, best_fixed_a].mean()):.5f} |",
        f"| Always GPT-4 | {100 * gpt4:.2f} | {float(C_eval[:, gpt4_a].mean()) if gpt4_a is not None else float('nan'):.5f} |",
        f"| Oracle | {100 * oracle:.2f} | |",
        "",
        "## Per-arm stats (avg over seeds)",
        "",
        "| Arm | Pulls | Empirical mean | True test mean |",
        "| --- | --- | --- | --- |",
    ]
    for a in order_by_mean:
        true_m = 100 * pure_barp if a == n_models else 100 * float(fixed_means[a])
        emp = f"{100 * means_mean[a]:.2f}" if counts_mean[a] > 0 else "--"
        md.append(f"| {arm_names[a]} | {counts_mean[a]:.1f} | {emp} | {true_m:.2f} |")
    out_md = args.checkpoint.parent / f"{stem}.md"
    out_md.write_text("\n".join(md) + "\n")
    print(f"Wrote {out_md}")

    # ----- figure: running average quality -----
    if not args.no_figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ts = np.arange(1, T + 1)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ts, 100 * curve_mean, label=f"online {args.greedy}-greedy", color="C0")
        ax.fill_between(ts, 100 * (curve_mean - curve_std), 100 * (curve_mean + curve_std),
                        alpha=0.2, color="C0")
        ax.axhline(100 * pure_barp, ls="--", color="C1", label="pure BaRP")
        ax.axhline(100 * best_fixed, ls="--", color="C2",
                   label=f"best fixed ({env.models[best_fixed_a].split('/')[-1]})")
        ax.axhline(100 * oracle, ls=":", color="gray", label="oracle")
        ax.set_xlabel("online step t")
        ax.set_ylabel("running avg quality (0-100)")
        ax.set_title(f"eps_t-greedy ({args.greedy} greedy) on {spec_name}  "
                     f"eps_t=min({args.eps_max:g},{eps_c:g}/t)")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out_png = args.checkpoint.parent / f"{stem}.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"Wrote {out_png}")

    # ----- W&B -----
    wb = maybe_resume_wandb(
        enabled=args.wandb,
        project=args.wandb_project,
        run_id=args.wandb_run_id,
        run_name=None,
        entity=args.wandb_entity,
        run_dir=args.checkpoint.parent,
        config={
            "online/greedy": args.greedy,
            "online/eps_c": eps_c,
            "online/eps_max": args.eps_max,
            "online/w_c": args.w_c,
            "online/seeds": args.seeds,
        },
    )
    if wb is not None:
        wb.log({
            f"online/{args.greedy}/avg_quality": 100 * final_avg.mean(),
            f"online/{args.greedy}/avg_quality_std": 100 * final_avg.std(),
            f"online/{args.greedy}/avg_cost": final_cost.mean(),
            "online/pure_barp": 100 * pure_barp,
            "online/best_fixed": 100 * best_fixed,
            "online/oracle": 100 * oracle,
        })
    wandb_finish(wb)


if __name__ == "__main__":
    main()
