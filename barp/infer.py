"""Per-prompt inference CLI for the trained BaRP router.

Two input modes (mutually exclusive):

    --prompt "<text>"       encode the prompt on the fly with MPNet (~5s import)
    --prompt-index N        use the cached embedding at row N of data/X.npy

In either mode you supply --w-q and --w-c (must sum to 1) and the trained
policy returns its preferred LLM along with the full softmax distribution.

When --prompt-index is used, ground-truth quality and cost from RouterBench
are also displayed so you can see how the policy's choice compares to the
best-quality and cheapest LLMs for that specific prompt.

Examples:
    # Raw prompt, balanced preference
    python -m barp.infer --checkpoint runs/barp/<ts>/policy.pt \\
        --prompt "What is the capital of France?" --w-q 0.5 --w-c 0.5

    # Cached prompt index 100, sweep three preferences with a shell loop:
    for wc in 0.0 0.5 1.0; do
        python -m barp.infer --checkpoint runs/barp/<ts>/policy.pt \\
            --prompt-index 100 --w-q $(python -c "print(1-$wc)") --w-c $wc
    done
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .env import RouterBenchBandit
from .model import BaRP
from .utils import pick_device


# Must match the encoder used to build data/X.npy (768-dim, frozen).
MPNET_MODEL = "sentence-transformers/all-mpnet-base-v2"


def encode_prompt(prompt: str, device: torch.device) -> np.ndarray:
    """Encode a single prompt to a 768-d MPNet embedding (lazy import)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise SystemExit(
            "sentence-transformers is required for --prompt mode. Install with "
            "`pip install sentence-transformers`, or use --prompt-index instead."
        ) from e
    encoder = SentenceTransformer(MPNET_MODEL, device=str(device))
    emb = encoder.encode([prompt], convert_to_numpy=True, normalize_embeddings=False)
    return emb.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt", type=str, default=None,
                     help="raw prompt text; encoded on the fly with MPNet")
    src.add_argument("--prompt-index", type=int, default=None,
                     help="row index of data/X.npy; uses the cached embedding")
    parser.add_argument("--w-q", type=float, required=True, help="weight on quality")
    parser.add_argument("--w-c", type=float, required=True, help="weight on cost")
    parser.add_argument("--top-k", type=int, default=5,
                       help="number of LLMs to display in the distribution")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if not (0.0 <= args.w_q <= 1.0) or not (0.0 <= args.w_c <= 1.0):
        parser.error("w_q and w_c must each lie in [0, 1]")
    if abs(args.w_q + args.w_c - 1.0) > 1e-6:
        parser.error(f"w_q + w_c must equal 1 (got {args.w_q + args.w_c:.6f})")

    device = pick_device(args.device)
    env = RouterBenchBandit(args.data_dir)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt["args"]

    model = BaRP(
        embed_dim=env.embed_dim,
        n_actions=env.n_actions,
        pref_dim=2,
        pref_hidden=ckpt_args.get("pref_hidden", 256),
        pref_out=ckpt_args.get("pref_out", 768),
        head_hidden=ckpt_args.get("head_hidden", 256),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # 1) Resolve the prompt embedding h
    if args.prompt is not None:
        h_np = encode_prompt(args.prompt, device)
        prompt_label = f'"{args.prompt[:80]}{"..." if len(args.prompt) > 80 else ""}"'
        prompt_source = f"encoded on the fly with {MPNET_MODEL}"
        cached_idx = None
    else:
        if not (0 <= args.prompt_index < env.X.shape[0]):
            parser.error(f"--prompt-index must be in [0, {env.X.shape[0]}); got {args.prompt_index}")
        h_np = np.asarray(env.X[args.prompt_index : args.prompt_index + 1], dtype=np.float32).copy()
        prompt_label = f"(cached embedding row {args.prompt_index})"
        prompt_source = f"cached embedding row {args.prompt_index} of X.npy"
        cached_idx = args.prompt_index

    h = torch.from_numpy(h_np).to(device)
    w = torch.tensor([[args.w_q, args.w_c]], dtype=torch.float32, device=device)

    # 2) Forward pass
    with torch.no_grad():
        logits = model(h, w)
        probs = logits.softmax(-1).cpu().numpy()[0]

    chosen = int(probs.argmax())
    chosen_name = env.models[chosen]

    print()
    print("BaRP per-prompt inference")
    print("=" * 64)
    print(f"checkpoint:    {args.checkpoint}")
    print(f"preference:    w_q={args.w_q:.2f}  w_c={args.w_c:.2f}")
    print(f"prompt:        {prompt_label}")
    print(f"prompt source: {prompt_source}")
    print()
    print(f"Routed to:     {chosen_name}   (confidence {probs[chosen]:.3f})")

    # 3) If we know the row in X, also show ground-truth q and c
    if cached_idx is not None:
        q_row = np.asarray(env.Q[cached_idx], dtype=np.float32)
        c_row = np.asarray(env.C[cached_idx], dtype=np.float32)
        best_q = int(q_row.argmax())
        cheapest = int(c_row.argmin())
        print()
        print("Ground truth on this prompt:")
        print(f"  policy's choice ({chosen_name}):")
        print(f"    q={q_row[chosen]:.3f}   c=${c_row[chosen]:.5f}")
        print(f"  best-quality LLM ({env.models[best_q]}):")
        print(f"    q={q_row[best_q]:.3f}   c=${c_row[best_q]:.5f}")
        print(f"  cheapest LLM ({env.models[cheapest]}):")
        print(f"    q={q_row[cheapest]:.3f}   c=${c_row[cheapest]:.5f}")

    # 4) Top-K distribution
    top_k = min(args.top_k, env.n_actions)
    order = np.argsort(-probs)[:top_k]
    name_w = max(len(m) for m in env.models)
    print()
    print(f"Top {top_k} LLMs (softmax probabilities):")
    for j in order:
        marker = "  <-- chosen" if j == chosen else ""
        print(f"  {probs[j]*100:5.1f}%  {env.models[j]:<{name_w}s}{marker}")


if __name__ == "__main__":
    main()
