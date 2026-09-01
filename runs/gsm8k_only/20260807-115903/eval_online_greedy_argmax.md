# Online eps_t-greedy (argmax-greedy) on `ood_hard_gsm8k`

- **stream:** test split, N = 7,450, T = 7,450, seeds = [0, 1, 2, 3, 4]
- **eps_t:** min(1, 12/t)   |   **arms:** 11 models + frozen BaRP (w_c=0.00)
- **checkpoint:** `runs/gsm8k_only/20260807-115903/policy.pt`

| Policy | Avg quality | Avg cost ($) |
| --- | --- | --- |
| Online argmax-greedy | 65.92 ± 0.20 | 0.00695 |
| Pure BaRP | 60.48 | 0.00053 |
| Best fixed (claude-v2) | 66.26 | 0.00604 |
| Always GPT-4 | 65.88 | 0.00855 |
| Oracle router | 74.94 | |

## Per-arm stats (avg over seeds)

| Arm | Pulls | Empirical mean | True test mean |
| --- | --- | --- | --- |
| gpt-4-1106-preview | 2924.2 | 64.12 | 65.88 |
| BaRP (w_c=0.00) | 34.6 | 62.00 | 60.48 |
| claude-v2 | 4409.6 | 61.18 | 66.26 |
| claude-instant-v1 | 11.0 | 59.32 | 62.72 |
| claude-v1 | 12.2 | 56.91 | 65.08 |
| gpt-3.5-turbo-1106 | 9.4 | 56.82 | 60.48 |
| zero-one-ai/Yi-34B-Chat | 9.8 | 53.93 | 54.81 |
| meta/llama-2-70b-chat | 7.4 | 48.95 | 52.30 |
| WizardLM/WizardLM-13B-V1.2 | 9.0 | 48.25 | 50.63 |
| meta/code-llama-instruct-34b-chat | 8.0 | 44.83 | 45.66 |
| mistralai/mistral-7b-chat | 9.2 | 44.58 | 41.15 |
| mistralai/mixtral-8x7b-chat | 5.6 | 38.21 | 51.90 |
