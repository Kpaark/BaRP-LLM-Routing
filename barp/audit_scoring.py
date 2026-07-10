"""Audit the scoring criteria of each RouterBench component.

Different benchmarks score "success" differently: some are strictly binary
(exact-match right/wrong), others award fractional credit (rubric or partial
scores). Comparing per-family quality numbers is only honest if we know which
scale each family uses -- this script produces that reference.

For every task family (and every eval_name inside non-binary families) it
reports, over all (prompt, model) pairs:

    n_pairs        number of (prompt, model) scores
    n_prompts      distinct prompts
    pct_zero/one   share of scores exactly 0.0 / exactly 1.0
    pct_frac       share strictly between 0 and 1 (partial credit)
    n_unique       distinct score values
    min/max/mean   score range
    scoring        "binary" (only {0,1}), "mostly_binary" (<1% fractional),
                   or "fractional"

Outputs:
    reports/scoring_audit_family.csv
    reports/scoring_audit_eval_name.csv
    printed summary table

Usage:
    python -m barp.audit_scoring
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PKL = REPO_ROOT.parent / "RouterBench_stats" / "data" / "routerbench_raw.pkl"
DEFAULT_IDS = REPO_ROOT.parent / "routerbench_gmm" / "data" / "embeddings.ids.csv"


def classify(pct_frac: float) -> str:
    if pct_frac == 0.0:
        return "binary"
    if pct_frac < 1.0:
        return "mostly_binary"
    return "fractional"


def audit(df: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for name, sub in df.groupby(by):
        perf = sub["performance"].to_numpy(dtype=np.float64)
        n = len(perf)
        pct_zero = float((perf == 0.0).mean() * 100)
        pct_one = float((perf == 1.0).mean() * 100)
        pct_frac = float(((perf > 0.0) & (perf < 1.0)).mean() * 100)
        rows.append({
            by: name,
            "n_pairs": n,
            "n_prompts": int(sub["sample_id"].nunique()),
            "pct_zero": round(pct_zero, 2),
            "pct_one": round(pct_one, 2),
            "pct_frac": round(pct_frac, 2),
            "n_unique": int(np.unique(perf).size),
            "min": float(perf.min()),
            "max": float(perf.max()),
            "mean": round(float(perf.mean()), 4),
            "scoring": classify(pct_frac),
        })
    return pd.DataFrame(rows).sort_values(by).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", type=Path, default=DEFAULT_PKL)
    parser.add_argument("--ids", type=Path, default=DEFAULT_IDS)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "reports")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.pkl}")
    raw = pd.read_pickle(args.pkl)[["sample_id", "eval_name", "model_name", "performance"]]
    raw["performance"] = raw["performance"].astype(float)

    fam_map = pd.read_csv(args.ids)[["sample_id", "family"]]
    df = raw.merge(fam_map, on="sample_id", how="left")
    n_unmapped = int(df["family"].isna().sum())
    if n_unmapped:
        print(f"warning: {n_unmapped:,} rows without a family mapping (dropped)")
        df = df.dropna(subset=["family"])
    print(f"{len(df):,} (prompt, model) scores across {df['sample_id'].nunique():,} prompts")

    # Family-level audit (the headline table).
    fam = audit(df, "family")
    fam_path = args.out_dir / "scoring_audit_family.csv"
    fam.to_csv(fam_path, index=False)

    cols = ["family", "n_prompts", "pct_zero", "pct_one", "pct_frac", "n_unique", "min", "max", "scoring"]
    print("\n=== Scoring criteria by task family ===")
    print(fam[cols].to_string(index=False))

    # Fine-grained audit for every eval_name inside non-binary families,
    # so we can see exactly which sub-benchmarks award partial credit.
    non_binary = fam.loc[fam["scoring"] != "binary", "family"].tolist()
    ev = audit(df[df["family"].isin(non_binary)], "eval_name")
    ev_path = args.out_dir / "scoring_audit_eval_name.csv"
    ev.to_csv(ev_path, index=False)

    if non_binary:
        print(f"\n=== eval_name detail for non-binary families {non_binary} ===")
        show = ev[ev["scoring"] != "binary"]
        cols_ev = ["eval_name", "n_prompts", "pct_frac", "n_unique", "min", "max", "scoring"]
        print(show[cols_ev].to_string(index=False))

    print(f"\nWrote {fam_path}")
    print(f"Wrote {ev_path}")


if __name__ == "__main__":
    main()
