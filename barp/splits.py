"""Split specification: the single source of truth for ID/OOD experiments.

A SplitSpec says which RouterBench task families go into train and which into
test. The same mechanism covers both experiment types:

    * In-distribution (ID):  a family in BOTH train and test is split
      within-family by prompt (test_frac to test, val_frac to val, rest train).
    * Out-of-distribution:   a family ONLY in test goes entirely to test;
      the router never sees it during training.
    * A family only in train is partitioned into train/val (no test rows).
    * A family in neither list is excluded from the experiment entirely.

Specs can be given inline on the CLI or as a JSON config, e.g.
experiments/ood_mbpp_hellaswag.json:

    {
      "name": "ood_mbpp_hellaswag",
      "train_families": ["ARC-Challenge", "GSM8K", "MMLU",
                         "MT-Bench", "RAG", "Winogrande"],
      "test_families": ["MBPP", "HellaSwag"],
      "val_frac": 0.15,
      "test_frac": 0.15,
      "seed": 42
    }

"all" (or omitting the key) means every family in the dataset. For
train_families, "rest" means every family NOT listed in test_families --
the usual way to write an OOD holdout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ALL = "all"
REST = "rest"  # train_families only: every family not in test_families


@dataclass
class SplitSpec:
    name: str = "id_full"
    train_families: list[str] | str = ALL
    test_families: list[str] | str = ALL
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 42

    @classmethod
    def from_file(cls, path: Path) -> "SplitSpec":
        cfg = json.loads(Path(path).read_text())
        unknown = set(cfg) - set(cls.__dataclass_fields__)
        if unknown:
            raise KeyError(f"{path}: unknown split-spec keys {sorted(unknown)}")
        return cls(**cfg)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "train_families": self.train_families,
            "test_families": self.test_families,
            "val_frac": self.val_frac,
            "test_frac": self.test_frac,
            "seed": self.seed,
        }

    def resolve(self, all_families: list[str]) -> tuple[list[str], list[str]]:
        """Expand "all"/"rest" and validate family names against the dataset."""
        def expand(fams: list[str] | str) -> list[str]:
            if fams == ALL:
                return sorted(all_families)
            missing = set(fams) - set(all_families)
            if missing:
                raise ValueError(
                    f"spec '{self.name}': families not in data: {sorted(missing)}; "
                    f"available: {sorted(all_families)}"
                )
            return sorted(fams)

        test = expand(self.test_families)
        if self.train_families == REST:
            train = sorted(set(all_families) - set(test))
        else:
            train = expand(self.train_families)
        return train, test

    def is_ood(self, all_families: list[str]) -> bool:
        train, test = self.resolve(all_families)
        return bool(set(test) - set(train))


def make_splits(families: np.ndarray, spec: SplitSpec) -> dict[str, list[int]]:
    """Assign every prompt index to train / val / test (or drop it) per spec.

    Per family (processed in sorted order for reproducibility):
      * in train and test: shuffled, then test_frac -> test, val_frac -> val,
        remainder -> train (the ID case).
      * only in test:      all rows -> test, untouched by the rng (OOD case).
      * only in train:     shuffled, val_frac -> val, remainder -> train.
      * in neither:        excluded.
    """
    families = np.asarray(families)
    all_fams = sorted(np.unique(families).tolist())
    train_fams, test_fams = spec.resolve(all_fams)

    rng = np.random.default_rng(spec.seed)
    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    all_idx = np.arange(len(families))

    for fam in all_fams:
        in_train, in_test = fam in train_fams, fam in test_fams
        if not in_train and not in_test:
            continue
        idx = all_idx[families == fam].copy()
        if not in_train:                       # OOD family: everything to test
            splits["test"].extend(idx.tolist())
            continue
        rng.shuffle(idx)
        n_test = int(round(len(idx) * spec.test_frac)) if in_test else 0
        n_val = int(round(len(idx) * spec.val_frac))
        splits["test"].extend(idx[:n_test].tolist())
        splits["val"].extend(idx[n_test : n_test + n_val].tolist())
        splits["train"].extend(idx[n_test + n_val :].tolist())

    for k in splits:
        splits[k].sort()
    return splits


def read_split_mode(data_dir: Path) -> str:
    """"ood" or "in_distribution", as recorded by build_bandit_table.py."""
    meta_path = Path(data_dir) / "meta.json"
    if not meta_path.exists():
        return "in_distribution"
    meta = json.loads(meta_path.read_text())
    mode = meta.get("split_mode")
    if mode:
        return mode
    return "ood" if meta.get("ood_families") else "in_distribution"


def load_spec_from_data_dir(data_dir: Path) -> SplitSpec | None:
    """Recover the spec recorded by build_bandit_table.py, if present."""
    meta_path = Path(data_dir) / "meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    if "split_spec" in meta:
        return SplitSpec(**meta["split_spec"])
    # Data dirs built before this refactor: reconstruct the closest spec.
    if meta.get("ood_families"):
        return SplitSpec(
            name="legacy_ood",
            train_families=REST,
            test_families=meta["ood_families"],
            val_frac=meta.get("val_frac", 0.15),
            test_frac=0.0,
            seed=meta.get("seed", 42),
        )
    return SplitSpec(
        name="legacy_id",
        val_frac=meta.get("val_frac", 0.15),
        test_frac=meta.get("test_frac", 0.15),
        seed=meta.get("seed", 42),
    )
