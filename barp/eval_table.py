"""Per-family results table for ID and OOD experiments (paper Tables 2 / 3).

Evaluates a Version B checkpoint on the test split of any data dir built by
`build_bandit_table.py` and reports mean quality **per task family**. The data
dir's meta.json says whether the test set is in-distribution (Table 2 style)
or held-out families (Table 3 style); the report is labeled accordingly.

Reference rows:
    * Smallest LLM  -- always pick the cheapest model (Mistral-7B)
    * Largest LLM   -- always pick the strongest model (GPT-4)
    * Oracle        -- per-prompt argmax quality (upper bound)
plus the BaRP policy evaluated at several preference points.

Usage:
    # In-distribution (Table 2 style)
    python -m barp.eval_table --data-dir data --checkpoint runs/barp/<ts>/policy.pt

    # OOD (Table 3 style)
    python -m barp.eval_table --data-dir data_ood --checkpoint runs/barp_ood/<ts>/policy.pt

    # Log to the same W&B run as training:
    python -m barp.eval_table --data-dir data --checkpoint <...>/policy.pt --wandb
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


# Preference points to evaluate. The paper reports single settings; we sweep
# several so you can see how quality trades against w_c.
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
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
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

    meta_path = args.data_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    split_mode = meta.get("split_mode") or ("ood" if meta.get("ood_families") else "in_distribution")
    spec_name = (meta.get("split_spec") or {}).get("name", args.data_dir.name)

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
            "eval_split_mode": split_mode,
            "eval_spec_name": spec_name,
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

    rows.append(("Oracle", per_family_quality(Q_eval, families_test, Q_eval.argmax(-1))))

    # ----- BaRP policy at several preference points -----
    for w_c in args.w_c:
        a = policy_actions(model, env, eval_idx, w_c, device)
        label = f"BaRP (Ours)  w_c={w_c:.2f}"
        rows.append((label, per_family_quality(Q_eval, families_test, a)))

    # ----- print Table 2/3 style report -----
    mode_label = "OOD" if split_mode == "ood" else "in-distribution"
    print(f"\n{mode_label} evaluation  [spec: {spec_name}]  split = test  (N = {len(eval_idx):,})")
    print(f"test families: {fam_order}")
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

    out_path = args.checkpoint.parent / f"eval_{split_mode}.json"
    out_path.write_text(json.dumps({
        "split": "test",
        "split_mode": split_mode,
        "spec_name": spec_name,
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
