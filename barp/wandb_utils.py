"""Optional Weights & Biases logging for BaRP training and evaluation.

All functions are no-ops unless ``--wandb`` is passed to the caller.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


class WandbSession:
    """Thin wrapper so train/eval code can log without importing wandb everywhere."""

    def __init__(self, run: Any) -> None:
        self.run = run
        self.run_id: str = run.id
        self.run_name: str = run.name
        self.project: str = run.project

    def log(self, metrics: dict[str, float | int], step: int | None = None) -> None:
        self.run.log(metrics, step=step)

    def log_table(self, key: str, table: Any) -> None:
        self.run.log({key: table})

    def finish(self) -> None:
        self.run.finish()


def _configure_wandb_dirs(run_dir: Path) -> None:
    wandb_dir = run_dir / "wandb"
    wandb_data = run_dir / "wandb_data"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    wandb_data.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_DIR", str(wandb_dir))
    os.environ.setdefault("WANDB_DATA_DIR", str(wandb_data))


def maybe_init_wandb(
    *,
    enabled: bool,
    project: str,
    run_name: str | None,
    entity: str | None,
    config: dict[str, Any],
    run_dir: Path,
) -> WandbSession | None:
    """Start a W&B run and persist the run id for later eval attachment."""
    if not enabled:
        return None

    import wandb

    _configure_wandb_dirs(run_dir)

    run = wandb.init(
        project=project,
        name=run_name,
        entity=entity or None,
        config=config,
        dir=str(run_dir),
    )
    session = WandbSession(run)
    (run_dir / "wandb_run_id.txt").write_text(session.run_id)
    print(f"wandb run: {session.project}/{session.run_name}  id={session.run_id}")
    return session


def resolve_wandb_run_id(run_id: str | None, run_dir: Path) -> str:
    """Read a saved run id from ``run_dir/wandb_run_id.txt`` when not passed."""
    if run_id:
        return run_id
    run_id_path = run_dir / "wandb_run_id.txt"
    if run_id_path.exists():
        return run_id_path.read_text().strip()
    raise ValueError(
        "W&B eval logging needs --wandb-run-id or a wandb_run_id.txt file "
        f"next to the checkpoint (looked in {run_dir})"
    )


def maybe_resume_wandb(
    *,
    enabled: bool,
    project: str,
    run_id: str | None,
    run_name: str | None,
    entity: str | None,
    run_dir: Path,
    config: dict[str, Any] | None = None,
) -> WandbSession | None:
    """Attach to an existing W&B run (typically after training) to log test metrics."""
    if not enabled:
        return None

    import wandb

    resolved_id = resolve_wandb_run_id(run_id, run_dir)
    _configure_wandb_dirs(run_dir)

    run = wandb.init(
        project=project,
        id=resolved_id,
        name=run_name,
        entity=entity or None,
        resume="must",
        config=config or {},
        dir=str(run_dir),
    )
    session = WandbSession(run)
    print(f"wandb resumed: {session.project}/{session.run_name}  id={session.run_id}")
    return session


def _method_to_metric_prefix(method: str) -> str:
    if method == "Smallest LLM":
        return "baseline/smallest_llm"
    if method == "Largest LLM":
        return "baseline/largest_llm"
    if method.startswith("BaRP (Ours)"):
        # "BaRP (Ours)  w_c=0.50" -> test/barp_wc0.50
        wc = method.split("w_c=")[-1].strip()
        return f"test/barp_wc{wc}"
    safe = method.lower().replace(" ", "_").replace("(", "").replace(")", "")
    return f"test/{safe}"


def log_test_results(
    session: WandbSession | None,
    *,
    fam_order: list[str],
    rows: list[dict[str, float | str]],
    primary_w_c: float | None = 0.5,
) -> None:
    """Log Table-3-style OOD test accuracies to W&B."""
    if session is None:
        return

    import wandb

    metrics: dict[str, float] = {}
    table_rows: list[list[str | float]] = []
    for row in rows:
        method = str(row["method"])
        prefix = _method_to_metric_prefix(method)
        vals = [float(row[fam]) for fam in fam_order]
        avg = float(row["Avg"])
        table_rows.append([method, *vals, avg])
        for fam, val in zip(fam_order, vals):
            fam_key = fam.replace("-", "_")
            metrics[f"{prefix}/{fam_key}"] = val
        metrics[f"{prefix}/avg"] = avg

        if primary_w_c is not None and method == f"BaRP (Ours)  w_c={primary_w_c:.2f}":
            for fam, val in zip(fam_order, vals):
                fam_key = fam.replace("-", "_")
                metrics[f"test/quality_{fam_key}"] = val
            metrics["test/quality_avg"] = avg

    session.log(metrics)

    table = wandb.Table(
        columns=["method", *fam_order, "Avg"],
        data=table_rows,
    )
    try:
        session.log_table("test_results", table)
    except Exception as exc:
        print(f"warning: could not log W&B test_results table ({exc})")


def log_split_labels(session: WandbSession | None, data_dir: Path) -> None:
    """Log train/val/test task-family composition (RouterBench family labels)."""
    if session is None:
        return

    import wandb

    splits_path = data_dir / "splits.json"
    fams_path = data_dir / "families.npy"
    if not splits_path.exists():
        print(f"warning: {splits_path} missing; skipping W&B split label logging")
        return

    splits = json.loads(splits_path.read_text())
    if fams_path.exists():
        families_all = np.load(fams_path, allow_pickle=True)
    else:
        print(f"warning: {fams_path} missing; skipping W&B split label logging")
        return

    rows: list[list[str | int | float]] = []
    split_summary: dict[str, list[str]] = {}
    for split_name in ("train", "val", "test"):
        indices = np.asarray(splits.get(split_name, []), dtype=np.int64)
        if len(indices) == 0:
            continue
        fams = families_all[indices]
        counts = Counter(str(f) for f in fams)
        split_summary[f"{split_name}_families"] = sorted(counts.keys())
        for fam, count in sorted(counts.items()):
            rows.append([split_name, fam, count, count / len(indices)])

    table = wandb.Table(
        columns=["split", "family", "count", "fraction"],
        data=rows,
    )
    try:
        session.log_table("split_labels", table)
    except Exception as exc:
        print(f"warning: could not log W&B split_labels table ({exc}); using config only")

    meta: dict[str, Any] = {}
    meta_path = data_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    session.run.config.update({
        "split_mode": "ood" if meta.get("ood_families") else "stratified",
        "ood_families": meta.get("ood_families"),
        **split_summary,
    })
    for split_name, fam, count, frac in rows:
        session.run.config[f"split/{split_name}/{fam}/count"] = count
        session.run.config[f"split/{split_name}/{fam}/fraction"] = round(frac, 4)


def log_train_step(
    session: WandbSession | None,
    step: int,
    *,
    loss: float,
    reward: float,
    quality: float,
    cost_usd: float,
    w_q: float,
    w_c: float,
    policy_entropy: float,
) -> None:
    if session is None:
        return
    session.log({
        "train/loss": loss,
        "train/reward": reward,
        "train/quality": quality,
        "train/cost_usd": cost_usd,
        "train/w_q": w_q,
        "train/w_c": w_c,
        "train/policy_entropy": policy_entropy,
    }, step=step)


def log_val_step(
    session: WandbSession | None,
    step: int,
    *,
    quality: float,
    cost_usd: float,
    action_entropy_bits: float,
) -> None:
    if session is None:
        return
    session.log({
        "val/quality": quality,
        "val/cost_usd": cost_usd,
        "val/action_entropy_bits": action_entropy_bits,
    }, step=step)


def finish(session: WandbSession | None) -> None:
    if session is not None:
        session.finish()
