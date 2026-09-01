# Online eps_t-greedy (barp-greedy) on `ood_hard_gsm8k`

- **stream:** test split, N = 7,450, T = 7,450, seeds = [0, 1, 2, 3, 4]
- **eps_t:** min(1, 12/t)   |   **arms:** 11 models + frozen BaRP (w_c=0.00)
- **checkpoint:** `runs/gsm8k_only/20260807-115903/policy.pt`

| Policy | Avg quality | Avg cost ($) |
| --- | --- | --- |
| Online barp-greedy | 60.40 ± 0.01 | 0.00054 |
| Pure BaRP | 60.48 | 0.00053 |
| Best fixed (claude-v2) | 66.26 | 0.00604 |
| Always GPT-4 | 65.88 | 0.00855 |
| Oracle router | 74.94 | |

## Per-arm stats (avg over seeds)

| Arm | Pulls | Empirical mean | True test mean |
| --- | --- | --- | --- |
| gpt-4-1106-preview | 7.4 | 66.04 | 65.88 |
| claude-v2 | 8.0 | 62.84 | 66.26 |
| claude-instant-v1 | 7.8 | 62.65 | 62.72 |
| BaRP (w_c=0.00) | 7368.8 | 60.46 | 60.48 |
| gpt-3.5-turbo-1106 | 8.4 | 58.00 | 60.48 |
| claude-v1 | 6.4 | 55.50 | 65.08 |
| zero-one-ai/Yi-34B-Chat | 6.8 | 55.12 | 54.81 |
| WizardLM/WizardLM-13B-V1.2 | 8.0 | 50.16 | 50.63 |
| meta/llama-2-70b-chat | 7.2 | 49.67 | 52.30 |
| meta/code-llama-instruct-34b-chat | 7.8 | 45.67 | 45.66 |
| mistralai/mistral-7b-chat | 8.4 | 45.35 | 41.15 |
| mistralai/mixtral-8x7b-chat | 5.0 | 40.86 | 51.90 |
