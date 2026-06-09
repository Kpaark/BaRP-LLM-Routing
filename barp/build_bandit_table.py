"""Build the offline bandit table consumed by both Version A and Version B.

Joins frozen MPNet embeddings with RouterBench per-(prompt, model) quality and
cost, splits stratified by task family, and writes everything under data/:

    X.npy, Q.npy, C.npy, models.json, splits.json, meta.json

Usage:
    python -m barp.build_bandit_table
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EMB = REPO_ROOT.parent / "routerbench_gmm" / "data" / "embeddings.npy"
DEFAULT_IDS = REPO_ROOT.parent / "routerbench_gmm" / "data" / "embeddings.ids.csv"
DEFAULT_PKL = REPO_ROOT.parent / "RouterBench_stats" / "data" / "routerbench_raw.pkl"
DEFAULT_OUT = REPO_ROOT / "data"


def stratified_split(
    families: pd.Series, val_frac: float, test_frac: float, seed: int,
) -> dict[str, list[int]]:
    """Stratify by task family so every split sees every family proportionally."""
    rng = np.random.default_rng(seed)
    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for _, idx in families.groupby(families).indices.items():
        idx = np.asarray(idx)
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_frac))
        n_val = int(round(len(idx) * val_frac))
        splits["test"].extend(idx[:n_test].tolist())
        splits["val"].extend(idx[n_test : n_test + n_val].tolist())
        splits["train"].extend(idx[n_test + n_val :].tolist())
    for k in splits:
        splits[k].sort()
    return splits


def ood_split(
    families: pd.Series, ood_families: list[str], val_frac: float, seed: int,
) -> dict[str, list[int]]:
    """OOD split: families in `ood_families` go entirely to test, all others
    are partitioned into train/val. Mirrors the paper's Table 3 setup."""
    rng = np.random.default_rng(seed)
    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    fam_arr = families.to_numpy()
    all_idx = np.arange(len(families))
    test_mask = np.isin(fam_arr, ood_families)
    splits["test"].extend(all_idx[test_mask].tolist())
    id_idx = all_idx[~test_mask]
    id_fams = fam_arr[~test_mask]
    for fam in np.unique(id_fams):
        idx = id_idx[id_fams == fam].copy()
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_frac))
        splits["val"].extend(idx[:n_val].tolist())
        splits["train"].extend(idx[n_val:].tolist())
    for k in splits:
        splits[k].sort()
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMB)
    parser.add_argument("--ids", type=Path, default=DEFAULT_IDS)
    parser.add_argument("--pkl", type=Path, default=DEFAULT_PKL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ood-families",
        nargs="+",
        default=None,
        help="If set, switch to OOD split: listed families go entirely to test, "
             "all others are partitioned into train/val. Replicates Table 3.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading embeddings from {args.embeddings}")
    X = np.load(args.embeddings).astype(np.float32)
    ids = pd.read_csv(args.ids)
    assert len(ids) == X.shape[0], "embeddings.npy and embeddings.ids.csv misaligned"
    print(f"  X: {X.shape}")

    print(f"Loading RouterBench pickle from {args.pkl}")
    raw = pd.read_pickle(args.pkl)
    missing = {"sample_id", "model_name", "performance", "cost"} - set(raw.columns)
    if missing:
        raise KeyError(f"pickle is missing columns: {missing}")
    print(f"  raw rows: {len(raw):,}")

    # Pivot long -> wide. Sorting the model names freezes the action ordering.
    models = sorted(raw["model_name"].unique().tolist())
    q_wide = raw.pivot_table(index="sample_id", columns="model_name", values="performance", aggfunc="mean")[models]
    c_wide = raw.pivot_table(index="sample_id", columns="model_name", values="cost", aggfunc="mean")[models]
    print(f"  {len(models)} actions (LLMs): {models}")

    # Keep only sample_ids present in both sources, preserving embedding row order.
    mask = ids["sample_id"].isin(q_wide.index).to_numpy()
    if not mask.all():
        print(f"  dropped {(~mask).sum():,} prompts missing from pickle")
    X = X[mask]
    ids = ids.loc[mask].reset_index(drop=True)

    Q = q_wide.reindex(ids["sample_id"]).to_numpy(dtype=np.float32)
    C = c_wide.reindex(ids["sample_id"]).to_numpy(dtype=np.float32)

    nan_q, nan_c = int(np.isnan(Q).sum()), int(np.isnan(C).sum())
    if nan_q or nan_c:
        print(f"  filling NaNs: Q={nan_q}, C={nan_c} (column means)")
        for col in range(len(models)):
            Q[np.isnan(Q[:, col]), col] = float(np.nanmean(Q[:, col]))
            C[np.isnan(C[:, col]), col] = float(np.nanmean(C[:, col]))

    if args.ood_families:
        missing = set(args.ood_families) - set(ids["family"].unique())
        if missing:
            raise ValueError(f"OOD families not found in data: {sorted(missing)}")
        splits = ood_split(ids["family"], args.ood_families, args.val_frac, args.seed)
        print(f"  OOD split: held-out families = {args.ood_families}")
    else:
        splits = stratified_split(ids["family"], args.val_frac, args.test_frac, args.seed)
    print(f"  splits: train={len(splits['train']):,} val={len(splits['val']):,} test={len(splits['test']):,}")
    test_fams = ids["family"].iloc[splits["test"]].value_counts()
    print(f"  test families:\n{test_fams.to_string()}")

    c_train = C[np.array(splits["train"])].ravel()
    tau_candidates = {
        "p50": float(np.percentile(c_train, 50)),
        "p75": float(np.percentile(c_train, 75)),
        "p95": float(np.percentile(c_train, 95)),
        "max": float(c_train.max()),
    }

    np.save(args.out_dir / "X.npy", X)
    np.save(args.out_dir / "Q.npy", Q)
    np.save(args.out_dir / "C.npy", C)
    np.save(args.out_dir / "families.npy", ids["family"].to_numpy())
    (args.out_dir / "models.json").write_text(json.dumps(models, indent=2))
    (args.out_dir / "splits.json").write_text(json.dumps(splits))
    (args.out_dir / "meta.json").write_text(json.dumps({
        "n_prompts": int(len(ids)),
        "n_actions": len(models),
        "embed_dim": int(X.shape[1]),
        "quality_mean": float(Q.mean()),
        "cost_mean_usd": float(C.mean()),
        "tau_candidates_usd": tau_candidates,
        "seed": args.seed,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "ood_families": args.ood_families,
    }, indent=2))
    print(f"Wrote bandit table to {args.out_dir}")


if __name__ == "__main__":
    main()
