"""One-command data setup for a fresh Mac (or any machine).

Downloads RouterBench from Hugging Face, encodes prompt embeddings, and builds
all experiment bandit tables under ``data/``, ``data_ood/``, etc.

Usage:
    # Full setup (~1.1 GB download + ~15-30 min encoding on Apple Silicon):
    python -m barp.setup_data --all

    # Only download + encode (skip bandit tables):
    python -m barp.setup_data --download --encode

    # Rebuild tables from an existing cache/ (skip download + encode):
    python -m barp.setup_data --tables
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .routerbench_io import (
    build_prompts_table,
    download_routerbench_pickle,
    encode_prompts,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = REPO_ROOT / "cache"

# (split config, output data dir) pairs used across the project.
TABLE_BUILDS: list[tuple[str, str]] = [
    ("experiments/id_full.json", "data"),
    ("experiments/ood_mbpp_hellaswag.json", "data_ood"),
    ("experiments/ood_hard_gsm8k.json", "data_ood_hard"),
    ("experiments/gsm8k_only.json", "data_gsm8k_only"),
]


def run_build_table(
    config: Path,
    out_dir: Path,
    *,
    embeddings: Path,
    ids: Path,
    pkl: Path,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "barp.build_bandit_table",
        "--config",
        str(config),
        "--out-dir",
        str(out_dir),
        "--embeddings",
        str(embeddings),
        "--ids",
        str(ids),
        "--pkl",
        str(pkl),
    ]
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE,
                        help="where to store the HF pickle and embeddings")
    parser.add_argument("--all", action="store_true",
                        help="download + encode + build all bandit tables")
    parser.add_argument("--download", action="store_true", help="fetch routerbench_raw.pkl")
    parser.add_argument("--encode", action="store_true", help="build MPNet embeddings")
    parser.add_argument("--tables", action="store_true", help="build all bandit tables")
    parser.add_argument("--encoder-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="cpu | mps | cuda (default: auto)")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-encode", action="store_true")
    args = parser.parse_args()

    if args.all:
        args.download = args.encode = args.tables = True
    if not (args.download or args.encode or args.tables):
        parser.error("pass --all or at least one of --download, --encode, --tables")

    cache = args.cache_dir
    pkl_path = cache / "routerbench_raw.pkl"
    prompts_csv = cache / "prompts.csv"
    embeddings_npy = cache / "embeddings.npy"
    embeddings_ids = cache / "embeddings.ids.csv"

    if args.download:
        if pkl_path.exists() and not args.force_download:
            print(f"Using existing pickle: {pkl_path}")
        else:
            print("Downloading RouterBench pickle from Hugging Face (~1.1 GB)...")
            pkl_path = download_routerbench_pickle(cache)
            print(f"Downloaded -> {pkl_path}")
        build_prompts_table(pkl_path, prompts_csv)

    if not pkl_path.exists():
        raise FileNotFoundError(
            f"{pkl_path} not found. Run with --download or --all first."
        )

    if args.encode:
        if embeddings_npy.exists() and embeddings_ids.exists() and not args.force_encode:
            print(f"Using existing embeddings: {embeddings_npy}")
        else:
            if not prompts_csv.exists():
                build_prompts_table(pkl_path, prompts_csv)
            encode_prompts(
                prompts_csv,
                embeddings_npy,
                model_name=args.encoder_model,
                batch_size=args.encode_batch_size,
                device=args.device,
            )

    if args.tables:
        if not embeddings_npy.exists() or not embeddings_ids.exists():
            raise FileNotFoundError(
                f"Missing {embeddings_npy} or {embeddings_ids}. "
                "Run with --encode or --all first."
            )
        for config_rel, out_rel in TABLE_BUILDS:
            config = REPO_ROOT / config_rel
            out_dir = REPO_ROOT / out_rel
            run_build_table(
                config,
                out_dir,
                embeddings=embeddings_npy,
                ids=embeddings_ids,
                pkl=pkl_path,
            )

    print("\nDone.")
    if args.tables:
        print("Bandit tables ready. Example train command:")
        print("  python -m barp.train --data-dir data --out-dir runs/id_full --steps 10000")


if __name__ == "__main__":
    main()
