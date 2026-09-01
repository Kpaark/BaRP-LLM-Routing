# Online eps_t-greedy (argmax-greedy) on `ood_hard_gsm8k`

- **stream:** test split, N = 7,450, T = 7,450, seeds = [0, 1, 2, 3, 4]
- **eps_t:** min(1, 2/t)   |   **arms:** 1 models + frozen BaRP (w_c=0.00)
- **checkpoint:** `runs/ood_hard_gsm8k/20260709-175836/policy.pt`

| Policy | Avg quality | Avg cost ($) |
| --- | --- | --- |
| Online argmax-greedy | 65.70 ± 1.06 | 0.00573 |
| Pure BaRP | 54.81 | 0.00039 |
| Best fixed (claude-v2) | 66.26 | 0.00604 |
| Always GPT-4 | 65.88 | 0.00855 |
| Oracle router | 74.94 | |

## Per-arm stats (avg over seeds)

| Arm | Pulls | Empirical mean | True test mean |
| --- | --- | --- | --- |
| claude-v2 | 7047.6 | 66.26 | 66.26 |
| BaRP (w_c=0.00) | 402.4 | 52.47 | 54.81 |
