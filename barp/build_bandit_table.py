"""Build the offline bandit table consumed by both Version A and Version B.

Joins frozen MPNet embeddings with RouterBench per-(prompt, model) quality and
cost, assigns prompts to train/val/test according to a SplitSpec (see
barp/splits.py), and writes everything under the output dir:

    X.npy, Q.npy, C.npy, families.npy, models.json, splits.json, meta.json

The split is fully described by a spec: which families are trained on and
which are tested on. Same code path covers ID and OOD experiments.

Usage:
    # In-distribution (default): every family in train AND test
    python -m barp.build_bandit_table

    # From a config file (recommended: one file = one experiment)
    python -m barp.build_bandit_table --config experiments/ood_mbpp_hellaswag.json --out-dir data_ood

    # Inline OOD: held-out families to test, the rest to train/val
    python -m barp.build_bandit_table --test-families MBPP HellaSwag --train-families rest --out-dir data_ood
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .splits import ALL, SplitSpec, make_splits


REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "cache"
# Self-contained defaults (populated by `python -m barp.setup_data --all`).
DEFAULT_EMB = CACHE_DIR / "embeddings.npy"
DEFAULT_IDS = CACHE_DIR / "embeddings.ids.csv"
DEFAULT_PKL = CACHE_DIR / "routerbench_raw.pkl"
DEFAULT_OUT = REPO_ROOT / "data"


def spec_from_args(args: argparse.Namespace) -> SplitSpec:
    if args.config is not None:
        spec = SplitSpec.from_file(args.config)
        print(f"split spec from {args.config}: {spec.to_dict()}")
        return spec
    train_fams = args.train_families if args.train_families else ALL
    test_fams = args.test_families if args.test_families else ALL
    if isinstance(train_fams, list) and len(train_fams) == 1 and train_fams[0] in (ALL, "rest"):
        train_fams = train_fams[0]
    if isinstance(test_fams, list) and len(test_fams) == 1 and test_fams[0] == ALL:
        test_fams = ALL
    name = args.name or ("id_full" if train_fams == ALL and test_fams == ALL else "custom")
    return SplitSpec(
        name=name,
        train_families=train_fams,
        test_families=test_fams,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMB)
    parser.add_argument("--ids", type=Path, default=DEFAULT_IDS)
    parser.add_argument("--pkl", type=Path, default=DEFAULT_PKL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--config", type=Path, default=None,
                        help="JSON split-spec file (overrides the flags below)")
    parser.add_argument("--name", default=None, help="experiment name for the spec")
    parser.add_argument("--train-families", nargs="+", default=None,
                        help="families to train on; 'all' (default) or 'rest' "
                             "(= everything not in --test-families)")
    parser.add_argument("--test-families", nargs="+", default=None,
                        help="families to test on; 'all' (default). A family in "
                             "test but not train is fully held out (OOD).")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    spec = spec_from_args(args)

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

    all_fams = sorted(ids["family"].unique().tolist())
    train_fams, test_fams_resolved = spec.resolve(all_fams)
    ood_fams = sorted(set(test_fams_resolved) - set(train_fams))
    split_mode = "ood" if ood_fams else "in_distribution"
    print(f"  spec '{spec.name}' ({split_mode})")
    print(f"    train families: {train_fams}")
    print(f"    test families:  {test_fams_resolved}" + (f"  (OOD: {ood_fams})" if ood_fams else ""))

    splits = make_splits(ids["family"].to_numpy(), spec)
    print(f"  splits: train={len(splits['train']):,} val={len(splits['val']):,} test={len(splits['test']):,}")
    test_fam_counts = ids["family"].iloc[splits["test"]].value_counts()
    print(f"  test families:\n{test_fam_counts.to_string()}")

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
        "split_spec": spec.to_dict(),
        "split_mode": split_mode,
        "train_families": train_fams,
        "test_families": test_fams_resolved,
        "ood_families": ood_fams or None,
    }, indent=2))
    print(f"Wrote bandit table to {args.out_dir}")


if __name__ == "__main__":
    main()
