# BaRP-LLM-Routing

A from-scratch implementation of **BaRP** (Bandit Routing with Preferences)
applied to the [RouterBench](https://huggingface.co/datasets/withmartian/routerbench)
corpus of 36,497 prompts and 11 candidate LLMs.

The router is trained as a contextual bandit: at each step it sees a prompt
embedding, picks one of 11 models, and observes the chosen model's quality
score and cost (table lookup against RouterBench, no live LLM calls).

This repository is built in two clearly-separated phases so that the
contribution of the preference encoder is visible in the git history:

1. **Phase 2 - BaRP without preference encoding.** A REINFORCE policy over
   prompt embeddings that maximizes raw quality. Establishes a baseline.
2. **Phase 3 - BaRP with preference MLP and cost-quality tradeoff.** Adds the
   preference encoder phi, samples preferences `w = (w_q, w_c)` on the
   1-simplex each step, and uses the cost-capped reward
   `r = w_q * q - w_c * min(c/tau, 1)`. This is the full Algorithm 1 below.

![Algorithm 1: BaRP training and inference](figures/algorithm1.png)

## Repository layout

```
BaRP_LLM_Routing/
  barp/                package code (models, env, trainers, eval)
  data/                cached tensors built by Phase 1 (gitignored)
  runs/                training logs / checkpoints (gitignored)
  figures/             plots committed to the repo (Algorithm 1, Pareto)
  scripts/             thin shell wrappers around python entry points
  requirements.txt
```

## Setup

```bash
cd BaRP_LLM_Routing
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This project assumes the sibling repos `../RouterBench_stats/` (provides the
raw RouterBench pickle) and `../routerbench_gmm/` (provides cached prompt
embeddings `data/embeddings.npy`) have already been run. See their READMEs
for one-time setup.

## Roadmap and reproduction commands

The commands below are **not wired up yet** -- they describe the target API
that each future PR will deliver. The scaffolding PR (this commit) only sets
up the package skeleton.

| Phase | PR | Command | Output |
|---|---|---|---|
| 1 | `data-pipeline` | `python -m barp.build_bandit_table` | `data/X.npy`, `data/Q.npy`, `data/C.npy`, `data/models.json`, `data/splits.json` |
| 2 | `barp-nopref` | `python -m barp.train_nopref --steps 10000` | `runs/nopref/<ts>/policy.pt` + metrics CSV |
| 3 | `barp-pref-mlp` | `python -m barp.train --steps 10000 --tau 0.01` | `runs/barp/<ts>/policy.pt` + metrics CSV |
| 3 | `barp-pref-mlp` | `python -m barp.eval_pareto` | `figures/pareto.png` |

## Mapping to Algorithm 1

| Line | Symbol | Implementation |
|---|---|---|
| 1 | encoder h | Frozen MPNet from `routerbench_gmm/encode_prompts.py`, loaded as cached `data/X.npy` |
| 1 | preference MLP phi | `barp.model.BaRP.phi` (Phase 3 only) |
| 1 | head g_theta | `barp.model.BaRP{NoPref,}.head` |
| 1 | cost cap tau | CLI flag `--tau` (Phase 3) |
| 1 | entropy coeff beta | CLI flag `--beta` |
| 4 | sample w on 1-simplex | `Dirichlet([1,1]).sample()` (Phase 3) |
| 5 | z = [h; u] | `torch.cat([h, phi(w)], dim=-1)` (Phase 3) |
| 6 | pi = softmax(o) | `logits.softmax(-1)` |
| 7 | a ~ Categorical(pi) | `torch.distributions.Categorical(pi).sample()` |
| 8 | observe q, c only for a | `barp.env.RouterBenchBandit.observe(a, idx)` |
| 9 | r = w_q q - w_c min(c/tau, 1) | reward in `barp.train` (Phase 3; reduces to `r = q` in Phase 2) |
| 10 | batch baseline b | `r.mean()` |
| 11 | loss with entropy bonus | `-((r - b).detach() * logp).mean() - beta * H.mean()` |
| 14 | inference: argmax pi | `barp.eval` |

## References

1. RouterBench: Hu et al., *A Benchmark for Multi-LLM Routing System*,
   arXiv:2403.12031, 2024.
2. BaRP paper: *citation pending - add when finalized*.
