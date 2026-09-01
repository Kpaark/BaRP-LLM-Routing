# Online eps_t-greedy (argmax-greedy) on `ood_hard_gsm8k`

- **stream:** test split, N = 7,450, T = 7,450, seeds = [0, 1, 2, 3, 4]
- **eps_t:** min(1, 12/t)   |   **arms:** 11 models + frozen BaRP (w_c=0.00)
- **checkpoint:** `runs/ood_hard_gsm8k_lr1e3/20260807-120502/policy.pt`

| Policy | Avg quality | Avg cost ($) |
| --- | --- | --- |
| Online argmax-greedy | 65.57 ± 0.85 | 0.00622 |
| Pure BaRP | 54.81 | 0.00039 |
| Best fixed (claude-v2) | 66.26 | 0.00604 |
| Always GPT-4 | 65.88 | 0.00855 |
| Oracle router | 74.94 | |

## Per-arm stats (avg over seeds)

| Arm | Pulls | Empirical mean | True test mean |
| --- | --- | --- | --- |
| gpt-4-1106-preview | 2172.0 | 64.15 | 65.88 |
| claude-v2 | 4412.0 | 61.18 | 66.26 |
| claude-instant-v1 | 782.4 | 60.12 | 62.72 |
| claude-v1 | 12.4 | 56.65 | 65.08 |
| gpt-3.5-turbo-1106 | 12.0 | 56.46 | 60.48 |
| BaRP (w_c=0.00) | 11.4 | 52.89 | 54.81 |
| zero-one-ai/Yi-34B-Chat | 8.4 | 52.60 | 54.81 |
| meta/llama-2-70b-chat | 7.4 | 48.95 | 52.30 |
| WizardLM/WizardLM-13B-V1.2 | 9.0 | 48.25 | 50.63 |
| mistralai/mistral-7b-chat | 9.4 | 44.93 | 41.15 |
| meta/code-llama-instruct-34b-chat | 8.0 | 44.83 | 45.66 |
| mistralai/mixtral-8x7b-chat | 5.6 | 38.21 | 51.90 |
