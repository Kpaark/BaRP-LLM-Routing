# OOD evaluation

- **spec:** `ood_hard_gsm8k`
- **split:** test (N = 7,450)
- **checkpoint:** `runs/ood_hard_gsm8k/20260709-175836/policy.pt`

| Method | GSM8K | Avg |
| --- | --- | --- |
| Smallest LLM | 41.2 | 41.2 |
| Largest LLM | 65.9 | 65.9 |
| Oracle | 74.9 | 74.9 |
| BaRP (Ours)  w_c=0.00 | 54.8 | 54.8 |
| BaRP (Ours)  w_c=0.25 | 54.8 | 54.8 |
| BaRP (Ours)  w_c=0.50 | 54.8 | 54.8 |
| BaRP (Ours)  w_c=0.75 | 50.6 | 50.6 |
| BaRP (Ours)  w_c=1.00 | 41.2 | 41.2 |
