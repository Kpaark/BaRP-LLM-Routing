# Online eps_t-greedy (argmax-greedy) on `ood_hard_gsm8k`

- **stream:** test split, N = 7,450, T = 7,450, seeds = [0, 1, 2, 3, 4]
- **eps_t:** min(1, 12/t)   |   **arms:** 11 models + frozen BaRP (w_c=0.00)
- **checkpoint:** `runs/ood_hard_gsm8k_beta05/20260807-120655/policy.pt`

| Policy | Avg quality | Avg cost ($) |
| --- | --- | --- |
| Online argmax-greedy | 65.70 ± 0.44 | 0.00695 |
| Pure BaRP | 65.88 | 0.00855 |
| Best fixed (claude-v2) | 66.26 | 0.00604 |
| Always GPT-4 | 65.88 | 0.00855 |
| Oracle router | 74.94 | |

## Per-arm stats (avg over seeds)

| Arm | Pulls | Empirical mean | True test mean |
| --- | --- | --- | --- |
| BaRP (w_c=0.00) | 1869.0 | 64.70 | 65.88 |
| gpt-4-1106-preview | 1570.0 | 63.90 | 65.88 |
| claude-v2 | 2557.4 | 61.07 | 66.26 |
| claude-instant-v1 | 14.8 | 59.27 | 62.72 |
| claude-v1 | 1377.6 | 57.52 | 65.08 |
| gpt-3.5-turbo-1106 | 13.2 | 56.67 | 60.48 |
| zero-one-ai/Yi-34B-Chat | 8.6 | 53.12 | 54.81 |
| meta/llama-2-70b-chat | 7.4 | 48.95 | 52.30 |
| WizardLM/WizardLM-13B-V1.2 | 9.0 | 48.25 | 50.63 |
| mistralai/mistral-7b-chat | 9.4 | 44.93 | 41.15 |
| meta/code-llama-instruct-34b-chat | 8.0 | 44.83 | 45.66 |
| mistralai/mixtral-8x7b-chat | 5.6 | 38.21 | 51.90 |
