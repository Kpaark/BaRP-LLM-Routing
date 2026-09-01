"""Download RouterBench and build the cached inputs used by build_bandit_table.

Everything lands under ``cache/`` inside the BaRP repo so collaborators do not
need the sibling ``RouterBench_stats`` or ``routerbench_gmm`` checkouts.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd


def download_routerbench_pickle(cache_dir: Path) -> Path:
    """Download routerbench_raw.pkl from Hugging Face (~1.1 GB, cached locally)."""
    from huggingface_hub import hf_hub_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id="withmartian/routerbench",
        filename="routerbench_raw.pkl",
        repo_type="dataset",
        local_dir=str(cache_dir),
    )
    return Path(path)


def unwrap_prompt(value) -> str:
    """RouterBench stores multi-turn prompts as stringified Python lists."""
    if isinstance(value, (list, tuple)):
        return "\n\n".join(str(v) for v in value)
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
            except (ValueError, SyntaxError):
                parsed = None
            if isinstance(parsed, (list, tuple)):
                return "\n\n".join(str(v) for v in parsed)
        return value
    return str(value)


def to_family(eval_name: str) -> str:
    name = eval_name.lower()
    if name.startswith("mmlu"):
        return "MMLU"
    if name.startswith("hellaswag"):
        return "HellaSwag"
    if name.startswith("grade-school-math") or "gsm" in name:
        return "GSM8K"
    if name.startswith("arc"):
        return "ARC-Challenge"
    if name.startswith("winogrande"):
        return "Winogrande"
    if name.startswith("mbpp"):
        return "MBPP"
    if name.startswith("mtbench"):
        return "MT-Bench"
    return "RAG"


def build_prompts_table(pkl_path: Path, out_csv: Path) -> pd.DataFrame:
    """One row per unique prompt with ``sample_id, prompt, eval_name, family``."""
    df = pd.read_pickle(pkl_path)
    prompts_df = (
        df[["sample_id", "prompt", "eval_name"]]
        .drop_duplicates(subset=["sample_id"])
        .reset_index(drop=True)
    )
    prompts_df["prompt"] = prompts_df["prompt"].map(unwrap_prompt)
    prompts_df["family"] = prompts_df["eval_name"].map(to_family)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    prompts_df.to_csv(out_csv, index=False)
    print(f"Saved {len(prompts_df):,} prompts -> {out_csv}")
    return prompts_df


def encode_prompts(
    prompts_csv: Path,
    out_npy: Path,
    *,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    batch_size: int = 32,
    device: str | None = None,
) -> tuple[np.ndarray, Path]:
    """Encode prompts with MPNet; write ``embeddings.npy`` and ``embeddings.ids.csv``."""
    import torch
    from sentence_transformers import SentenceTransformer

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    df = pd.read_csv(prompts_csv)
    print(f"Encoding {len(df):,} prompts with {model_name} on {device}")

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        df["prompt"].astype(str).tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, embeddings)
    ids_csv = out_npy.with_suffix(".ids.csv")
    df[["sample_id", "eval_name", "family"]].to_csv(ids_csv, index=False)
    print(f"Saved {embeddings.shape} -> {out_npy}")
    return embeddings, ids_csv
