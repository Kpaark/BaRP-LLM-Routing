#!/usr/bin/env bash
# One-shot setup for a fresh Mac clone of BaRP-LLM-Routing.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> Creating virtual environment"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Installing Python dependencies"
pip install --upgrade pip
pip install -r requirements.txt

echo "==> Downloading RouterBench, encoding prompts, building bandit tables"
echo "    (first run: ~1.1 GB download + ~15-30 min encoding on Apple Silicon)"
python -m barp.setup_data --all

echo
echo "Setup complete. Activate the venv and train:"
echo "  cd $ROOT"
echo "  source .venv/bin/activate"
echo "  python -m barp.train --data-dir data --out-dir runs/id_full --steps 10000"
