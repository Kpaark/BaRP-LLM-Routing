"""Mixture policy: blend BaRP with a fallback policy at fixed random probability.

For each test prompt, an independent (seeded) coin flip with bias p decides who
routes it:
    with prob 1-p  ->  BaRP's action (at preference w_c)
    with prob p    ->  the fallback policy's action

Sweeping p from 0 to 1 traces a curve from pure BaRP to the pure fallback.
This quantifies e.g. how much a "just use the most expensive model" fallback
repairs BaRP's hard-OOD failure, and at what dollar cost.

Fallback policies (--mix-policy):
    most-expensive   always the model with the highest mean cost on train (GPT-4)
    cheapest         always the model with the lowest mean cost on train
    best-single      always the model with the highest mean quality on train
    random           a uniformly random model per prompt

Usage:
    # Hard OOD (GSM8K): how much does a GPT-4 fallback repair the failure?
    python -m barp.eval_mixture --data-dir data_ood_hard \\
        --checkpoint runs/ood_hard_gsm8k/<ts>/policy.pt

    # In-distribution, custom sweep, logged to W&B:
    python -m barp.eval_mixture --data-dir data --checkpoint <...>/policy.pt \\
        --mix-probs 0 0.25 0.5 0.75 1.0 --wandb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .env import RouterBenchBandit
from .eval_table import per_family_quality, policy_actions
from .model import BaRP
from .utils import pick_device
from .wandb_utils import finish as wandb_finish
from .wandb_utils import maybe_resume_wandb

DEFAULT_MIX_PROBS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)


def fallback_actions(
    policy: str, env: RouterBenchBandit, n: int, rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    """Per-prompt actions for the fallback policy, plus a display label.

    Model selection stats use the TRAIN split only, so the fallback never
    peeks at test outcomes."""
    _, Q_train, C_train = env.qc_matrix("train")
    if policy == "most-expensive":
        a = int(np.argmax(C_train.mean(0)))
        return np.full(n, a, dtype=np.int64), f"most-expensive ({env.models[a]})"
    if policy == "cheapest":
        a = int(np.argmin(C_train.mean(0)))
        return np.full(n, a, dtype=np.int64), f"cheapest ({env.models[a]})"
    if policy == "best-single":
        a = int(np.argmax(Q_train.mean(0)))
        return np.full(n, a, dtype=np.int64), f"best-single ({env.models[a]})"
    if policy == "random":
        return rng.integers(0, env.n_actions, size=n), "random model"
    raise ValueError(f"unknown mix policy: {policy}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--w-c", type=float, default=0.0,
                        help="BaRP preference used inside the mixture "
                             "(default 0.0 = quality-focused)")
    parser.add_argument("--mix-policy", default="most-expensive",
                        choices=["most-expensive", "cheapest", "best-single", "random"])
    parser.add_argument("--mix-probs", type=float, nargs="+",
                        default=list(DEFAULT_MIX_PROBS),
                        help="fallback probabilities p to sweep (0 = pure BaRP)")
    parser.add_argument("--mix-seed", type=int, default=0,
                        help="seed for the per-prompt coin flips (reproducible)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="barp-llm-routing")
    parser.add_argument("--wandb-run-id", default=None)
    parser.add_argument("--wandb-entity", default=None)
    args = parser.parse_args()

    device = pick_device(args.device)
    env = RouterBenchBandit(args.data_dir)

    meta_path = args.data_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    split_mode = meta.get("split_mode") or ("ood" if meta.get("ood_families") else "in_distribution")
    spec_name = (meta.get("split_spec") or {}).get("name", args.data_dir.name)

    families_all = np.load(args.data_dir / "families.npy", allow_pickle=True)
    eval_idx, Q_eval, C_eval = env.qc_matrix("test")
    families_test = families_all[eval_idx]
    fam_order = sorted(np.unique(families_test).tolist())
    n = len(eval_idx)
    prompt_rows = np.arange(n)

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

    rng = np.random.default_rng(args.mix_seed)
    barp_a = policy_actions(model, env, eval_idx, args.w_c, device)
    fb_a, fb_label = fallback_actions(args.mix_policy, env, n, rng)

    # One coin flip per prompt, drawn once. Reusing the same uniforms across
    # the sweep makes the masks nested: prompts overridden at p=0.25 are a
    # subset of those overridden at p=0.5, so the curve is monotone in the
    # override set, not re-randomized at every p.
    coins = rng.random(n)

    rows: list[dict] = []

    def add_row(label: str, actions: np.ndarray) -> None:
        per_fam = per_family_quality(Q_eval, families_test, actions)
        vals = [per_fam.get(f, float("nan")) for f in fam_order]
        rows.append({
            "method": label,
            **{f: v for f, v in zip(fam_order, vals)},
            "Avg": float(np.mean(vals)),
            "Cost ($)": float(C_eval[prompt_rows, actions].mean()),
        })

    add_row(f"BaRP alone (w_c={args.w_c:.2f})", barp_a)
    add_row(f"Fallback alone: {fb_label}", fb_a)
    for p in args.mix_probs:
        mixed = np.where(coins < p, fb_a, barp_a)
        add_row(f"Mixture p={p:.2f}", mixed)

    # ----- print -----
    mode_label = "OOD" if split_mode == "ood" else "in-distribution"
    print(f"\nMixture evaluation ({mode_label})  [spec: {spec_name}]  split = test  (N = {n:,})")
    print(f"BaRP w_c={args.w_c:.2f}  +  fallback '{fb_label}'  (mix seed {args.mix_seed})")
    print(f"checkpoint: {args.checkpoint}")

    col_w = 14
    header = (f"{'Method':<38s}"
              + "".join(f"{f:>{col_w}s}" for f in fam_order)
              + f"{'Avg':>{col_w}s}{'Cost ($)':>{col_w}s}")
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        line = (f"{r['method']:<38s}"
                + "".join(f"{r[f]:>{col_w}.2f}" for f in fam_order)
                + f"{r['Avg']:>{col_w}.2f}{r['Cost ($)']:>{col_w}.5f}")
        print(line)

    # ----- save json + markdown -----
    stem = f"eval_mixture_{args.mix_policy.replace('-', '_')}"
    out_json = args.checkpoint.parent / f"{stem}.json"
    out_json.write_text(json.dumps({
        "split": "test",
        "split_mode": split_mode,
        "spec_name": spec_name,
        "n_test": n,
        "w_c": args.w_c,
        "mix_policy": args.mix_policy,
        "fallback": fb_label,
        "mix_seed": args.mix_seed,
        "families": fam_order,
        "rows": rows,
    }, indent=2))

    header_md = ["Method", *fam_order, "Avg", "Cost ($)"]
    md_lines = [
        f"# Mixture evaluation ({mode_label})",
        "",
        f"- **spec:** `{spec_name}`  |  **BaRP w_c:** {args.w_c:.2f}  |  "
        f"**fallback:** {fb_label}  |  N = {n:,}",
        "",
        "| " + " | ".join(header_md) + " |",
        "| " + " | ".join("---" for _ in header_md) + " |",
    ]
    for r in rows:
        cells = [str(r["method"])]
        cells += [f"{r[f]:.1f}" for f in fam_order]
        cells += [f"{r['Avg']:.1f}", f"{r['Cost ($)']:.5f}"]
        md_lines.append("| " + " | ".join(cells) + " |")
    out_md = args.checkpoint.parent / f"{stem}.md"
    out_md.write_text("\n".join(md_lines) + "\n")
    print(f"\nWrote {out_json}")
    print(f"Wrote {out_md}")

    # ----- W&B (attach to the training run) -----
    wb = maybe_resume_wandb(
        enabled=args.wandb,
        project=args.wandb_project,
        run_id=args.wandb_run_id,
        run_name=None,
        entity=args.wandb_entity,
        run_dir=args.checkpoint.parent,
        config={
            "mixture/mix_policy": args.mix_policy,
            "mixture/w_c": args.w_c,
            "mixture/mix_seed": args.mix_seed,
            "mixture/probs": args.mix_probs,
        },
    )
    if wb is not None:
        import wandb

        metrics: dict[str, float] = {}
        for r, p in zip(rows[2:], args.mix_probs):
            metrics[f"mixture/{args.mix_policy}/avg_p{p:.2f}"] = r["Avg"]
            metrics[f"mixture/{args.mix_policy}/cost_p{p:.2f}"] = r["Cost ($)"]
        wb.log(metrics)
        table = wandb.Table(
            columns=header_md,
            data=[[r["method"], *[r[f] for f in fam_order], r["Avg"], r["Cost ($)"]] for r in rows],
        )
        try:
            wb.log_table(f"mixture_{args.mix_policy}", table)
        except Exception as exc:
            print(f"warning: could not log W&B mixture table ({exc})")
    wandb_finish(wb)


if __name__ == "__main__":
    main()
