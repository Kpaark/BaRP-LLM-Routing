"""Select a *challenging* OOD holdout from prompt-embedding geometry.

Idea: an OOD test is hard when the held-out prompts are far from anything the
router saw in training. We measure that directly: project all prompt
embeddings to 2D with PCA, compute each task family's centroid, and rank
families by how far their centroid sits from the rest. The farthest family is
the maximally-OOD holdout.

Outputs:
    reports/centroid_distances.csv     pairwise centroid distances (2D + 768-d)
    figures/hard_ood_centroids.png     2D map: prompts, centroids, chosen holdout
    experiments/ood_hard_<family>.json ready-to-run split config
    printed ranking + recommendation

Usage:
    python -m barp.select_hard_ood
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Sandbox-safe matplotlib cache (same pattern as the gmm repo's scripts).
_MPL_CACHE = REPO_ROOT / ".mpl-cache"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(_MPL_CACHE))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_EMB = REPO_ROOT.parent / "routerbench_gmm" / "data" / "embeddings.npy"
DEFAULT_IDS = REPO_ROOT.parent / "routerbench_gmm" / "data" / "embeddings.ids.csv"


def centroid_table(X: np.ndarray, families: np.ndarray) -> tuple[list[str], np.ndarray]:
    fams = sorted(np.unique(families).tolist())
    cents = np.stack([X[families == f].mean(axis=0) for f in fams])
    return fams, cents


def pairwise(cents: np.ndarray) -> np.ndarray:
    diff = cents[:, None, :] - cents[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMB)
    parser.add_argument("--ids", type=Path, default=DEFAULT_IDS)
    parser.add_argument("--fig-dir", type=Path, default=REPO_ROOT / "figures")
    parser.add_argument("--report-dir", type=Path, default=REPO_ROOT / "reports")
    parser.add_argument("--exp-dir", type=Path, default=REPO_ROOT / "experiments")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for d in (args.fig_dir, args.report_dir, args.exp_dir):
        d.mkdir(parents=True, exist_ok=True)

    X = np.load(args.embeddings).astype(np.float32)
    ids = pd.read_csv(args.ids)
    families = ids["family"].to_numpy()

    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=args.seed)
    X2 = pca.fit_transform(X)
    print(f"PCA(2) explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    fams, cents2 = centroid_table(X2, families)
    _, cents_full = centroid_table(X, families)
    D2, Dfull = pairwise(cents2), pairwise(cents_full)

    # Hardness score: a family's distance to its NEAREST other family centroid.
    # (Far from the *closest* neighbor = nothing similar in training data.)
    rows = []
    for i, fam in enumerate(fams):
        others = [j for j in range(len(fams)) if j != i]
        rows.append({
            "family": fam,
            "nearest_2d": float(D2[i, others].min()),
            "nearest_family_2d": fams[others[int(D2[i, others].argmin())]],
            "mean_2d": float(D2[i, others].mean()),
            "nearest_768d": float(Dfull[i, others].min()),
            "mean_768d": float(Dfull[i, others].mean()),
        })
    rank = pd.DataFrame(rows).sort_values("nearest_2d", ascending=False).reset_index(drop=True)

    dist_df = pd.DataFrame(D2, index=fams, columns=fams)
    dist_path = args.report_dir / "centroid_distances.csv"
    dist_df.round(4).to_csv(dist_path)

    print("\n=== Families ranked by isolation (2D centroid distance to nearest neighbor) ===")
    print(rank.round(3).to_string(index=False))

    hard_fam = str(rank.iloc[0]["family"])
    print(f"\nHardest OOD holdout: {hard_fam} "
          f"(nearest neighbor {rank.iloc[0]['nearest_family_2d']} "
          f"at {rank.iloc[0]['nearest_2d']:.3f}; "
          f"full 768-d nearest {rank.iloc[0]['nearest_768d']:.3f})")

    # ---- 2D map ----
    rng = np.random.default_rng(args.seed)
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("tab10")
    for i, fam in enumerate(fams):
        pts = X2[families == fam]
        if len(pts) > 3000:                       # subsample for a readable plot
            pts = pts[rng.choice(len(pts), 3000, replace=False)]
        ax.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.25, color=cmap(i % 10),
                   label=fam, rasterized=True)
    for i, fam in enumerate(fams):
        edge = "red" if fam == hard_fam else "black"
        lw = 2.5 if fam == hard_fam else 1.0
        ax.scatter(*cents2[i], marker="X", s=260, color=cmap(i % 10),
                   edgecolor=edge, linewidth=lw, zorder=5)
        ax.annotate(fam, cents2[i], fontsize=10, fontweight="bold",
                    xytext=(6, 6), textcoords="offset points", zorder=6)
    ax.set_title(
        f"RouterBench prompt embeddings (PCA 2D) -- centroids marked;\n"
        f"hardest OOD holdout by isolation: {hard_fam} (red outline)"
    )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(markerscale=6, loc="best", fontsize=9)
    fig.tight_layout()
    fig_path = args.fig_dir / "hard_ood_centroids.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved {fig_path}")

    # ---- ready-to-run experiment config ----
    cfg = {
        "name": f"ood_hard_{hard_fam.lower().replace('-', '_')}",
        "train_families": "rest",
        "test_families": [hard_fam],
        "val_frac": 0.15,
        "test_frac": 0.0,
        "seed": 42,
    }
    cfg_path = args.exp_dir / f"{cfg['name']}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"Wrote {cfg_path}")
    print(f"Wrote {dist_path}")


if __name__ == "__main__":
    main()
