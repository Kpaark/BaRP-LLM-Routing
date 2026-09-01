"""Overlay online Quality-vs-Time curves from several eval_online runs.

Each eval_online run saves its running-average curve (plus reference curves)
in eval_online_greedy_<greedy>[_tag].json next to its checkpoint. This script
overlays the bandit curves of several such runs -- e.g. the same online
experiment repeated with different BaRP training variants -- in one figure.

Reference curves (pure BaRP / best fixed / oracle router) differ per variant
only through BaRP, so we draw best-fixed and oracle once (from the first run)
and label each variant's bandit curve with its name.

Usage:
    python -m barp.plot_online_comparison \\
        --runs current=runs/ood_hard_gsm8k/<ts>/eval_online_greedy_barp.json \\
               small=runs/ood_hard_gsm8k_small/<ts>/eval_online_greedy_barp.json \\
        --out figures/online_variants_barp.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True,
                        help="label=path/to/eval_online_*.json entries")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    first = None
    for k, entry in enumerate(args.runs):
        # Split at the LAST '=' so labels may contain '=' (e.g. "lr=1e-3").
        label, _, path = entry.rpartition("=")
        if not path:
            raise ValueError(f"--runs entries must be label=path, got: {entry}")
        data = json.loads(Path(path).read_text())
        if first is None:
            first = data
        curve = np.asarray(data["running_avg_mean"])
        std = np.asarray(data["running_avg_std"])
        ts = np.arange(1, len(curve) + 1)
        color = f"C{k}"
        final = data["online_avg_quality_mean"]
        ax.plot(ts, curve, color=color, label=f"{label} (final {final:.1f})")
        ax.fill_between(ts, curve - std, curve + std, alpha=0.15, color=color)

    # Shared references from the first run (identical streams across runs).
    refs = first.get("ref_running_avg", {})
    for name, curve in refs.items():
        if name.startswith("pure BaRP"):
            continue  # per-variant; the bandit curves already carry that info
        curve = np.asarray(curve)
        ts = np.arange(1, len(curve) + 1)
        style = ":" if "oracle" in name else "--"
        ax.plot(ts, curve, ls=style, color="gray", label=name)

    greedy = first.get("greedy", "?")
    ax.set_xlabel("online step t")
    ax.set_ylabel("running avg quality (0-100)")
    ax.set_title(args.title or
                 f"Online eps_t-greedy ({greedy} greedy) on {first.get('spec_name', '?')}: "
                 f"BaRP variants")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
