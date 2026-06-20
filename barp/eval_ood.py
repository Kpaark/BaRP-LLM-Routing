"""Replicate Table 3 of the BaRP paper: OOD per-task quality.

Evaluates a Version B checkpoint on the OOD test split (built with
`build_bandit_table.py --ood-families ...`) and reports mean quality
**per task family** so the numbers line up directly with the paper's
Table 3 (MBPP, HellaSwag, HpQA, Avg).

The output adds two reference rows from the paper:
    * Smallest LLM  -- always pick the cheapest model (Mistral-7B)
    * Largest LLM   -- always pick the strongest model (GPT-4)
plus our BaRP policy evaluated at several preference points.

Usage:
    python -m barp.eval_ood \\
        --data-dir data_ood \\
        --checkpoint runs/barp_ood/<ts>/policy.pt

    # Log test accuracy to the same W&B run as training:
    python -m barp.eval_ood \\
        --data-dir data_ood \\
        --checkpoint runs/wandb_smoke3/<ts>/policy.pt \\
        --wandb --wandb-project barp-llm-routing
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
from .wandb_utils import finish as wandb_finish
from .wandb_utils import log_test_results, maybe_resume_wandb


# Preference points to evaluate. The paper's Table 3 reports a single setting;
# we sweep several so you can see how OOD quality trades against w_c.
DEFAULT_W_C = (0.0, 0.25, 0.5, 0.75, 1.0)

# Paper-conventional "smallest" and "largest" LLMs in the RouterBench panel.
SMALLEST_LLM = "mistralai/mistral-7b-chat"
LARGEST_LLM = "gpt-4-1106-preview"


@torch.no_grad()
def policy_actions(
    model: BaRP, env: RouterBenchBandit, indices: np.ndarray, w_c: float,
    device: torch.device, batch_size: int = 1024,
) -> np.ndarray:
    model.eval()
    w_row = torch.tensor([[1.0 - w_c, w_c]], dtype=torch.float32, device=device)
    out = np.zeros(len(indices), dtype=np.int64)
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        h = torch.from_numpy(np.asarray(env.X[indices[start:stop]], dtype=np.float32)).to(device)
        w = w_row.expand(stop - start, -1)
        out[start:stop] = model(h, w).argmax(-1).cpu().numpy()
    return out


def per_family_quality(
    Q_eval: np.ndarray, families: np.ndarray, actions: np.ndarray,
) -> dict[str, float]:
    """Mean quality on each task family, expressed as a percentage (0-100)."""
    rows = np.arange(len(actions))
    chosen_q = Q_eval[rows, actions]
    out: dict[str, float] = {}
    for fam in np.unique(families):
        mask = families == fam
        if mask.any():
            out[str(fam)] = float(chosen_q[mask].mean() * 100.0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data_ood"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--w-c", type=float, nargs="+", default=list(DEFAULT_W_C))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wandb", action="store_true", help="log test metrics to Weights & Biases")
    parser.add_argument("--wandb-project", default="barp-llm-routing")
    parser.add_argument("--wandb-run-id", default=None,
                        help="attach to an existing run; defaults to checkpoint dir wandb_run_id.txt")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-primary-w-c", type=float, default=0.5,
                        help="w_c used for summary test/quality_* metrics")
    args = parser.parse_args()

    device = pick_device(args.device)
    env = RouterBenchBandit(args.data_dir)

    fams_path = args.data_dir / "families.npy"
    if not fams_path.exists():
        raise FileNotFoundError(
            f"{fams_path} not found. Rebuild data with the updated "
            f"build_bandit_table.py so families.npy is emitted."
        )
    families_all = np.load(fams_path, allow_pickle=True)

    eval_idx, Q_eval, _C_eval = env.qc_matrix("test")
    families_test = families_all[eval_idx]
    fam_order = sorted(np.unique(families_test).tolist())

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt["args"]
    run_dir = args.checkpoint.parent

    wb = maybe_resume_wandb(
        enabled=args.wandb,
        project=args.wandb_project,
        run_id=args.wandb_run_id,
        run_name=args.wandb_run_name,
        entity=args.wandb_entity,
        run_dir=run_dir,
        config={
            "eval_split": "test",
            "checkpoint": str(args.checkpoint),
            "data_dir": str(args.data_dir),
            "w_c_sweep": args.w_c,
        },
    )

    model = BaRP(
        embed_dim=env.embed_dim,
        n_actions=env.n_actions,
        pref_dim=2,
        pref_hidden=ckpt_args.get("pref_hidden", 256),
        pref_out=ckpt_args.get("pref_out", 768),
        head_hidden=ckpt_args.get("head_hidden", 256),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])

    # ----- reference rows -----
    rows: list[tuple[str, dict[str, float]]] = []

    for label, name in [("Smallest LLM", SMALLEST_LLM), ("Largest LLM", LARGEST_LLM)]:
        if name not in env.models:
            print(f"warning: {name} not in models list; skipping {label}")
            continue
        a = np.full(len(eval_idx), env.models.index(name), dtype=np.int64)
        rows.append((label, per_family_quality(Q_eval, families_test, a)))

    # ----- BaRP policy at several preference points -----
    for w_c in args.w_c:
        a = policy_actions(model, env, eval_idx, w_c, device)
        label = f"BaRP (Ours)  w_c={w_c:.2f}"
        rows.append((label, per_family_quality(Q_eval, families_test, a)))

    # ----- print Table 3 style report -----
    print(f"\nOOD evaluation on split = test  (N = {len(eval_idx):,})")
    print(f"held-out families: {fam_order}")
    print(f"checkpoint: {args.checkpoint}")

    col_w = 14
    header = f"{'Method':<26s}" + "".join(f"{fam:>{col_w}s}" for fam in fam_order) + f"{'Avg':>{col_w}s}"
    print("\n" + header)
    print("-" * len(header))
    table: list[dict] = []
    for label, per_fam in rows:
        vals = [per_fam.get(fam, float("nan")) for fam in fam_order]
        avg = float(np.mean(vals))
        line = f"{label:<26s}" + "".join(f"{v:>{col_w}.2f}" for v in vals) + f"{avg:>{col_w}.2f}"
        print(line)
        table.append({"method": label, **{fam: v for fam, v in zip(fam_order, vals)}, "Avg": avg})

    out_path = args.checkpoint.parent / "eval_ood.json"
    out_path.write_text(json.dumps({
        "split": "test",
        "n_test": int(len(eval_idx)),
        "families": fam_order,
        "rows": table,
    }, indent=2))
    print(f"\nWrote {out_path}")

    log_test_results(
        wb,
        fam_order=fam_order,
        rows=table,
        primary_w_c=args.wandb_primary_w_c,
    )
    wandb_finish(wb)


if __name__ == "__main__":
    main()
