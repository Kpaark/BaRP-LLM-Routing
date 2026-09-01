# OOD evaluation

- **spec:** `ood_mbpp_hellaswag`
- **split:** test (N = 10,469)
- **checkpoint:** `runs/ood/20260709-113603/policy.pt`

| Method | HellaSwag | MBPP | Avg |
| --- | --- | --- | --- |
| Smallest LLM | 53.7 | 34.4 | 44.1 |
| Largest LLM | 90.2 | 68.6 | 79.4 |
| Oracle | 98.4 | 86.7 | 92.5 |
| BaRP (Ours)  w_c=0.00 | 90.2 | 68.6 | 79.4 |
| BaRP (Ours)  w_c=0.25 | 87.5 | 44.7 | 66.1 |
| BaRP (Ours)  w_c=0.50 | 87.5 | 43.6 | 65.5 |
| BaRP (Ours)  w_c=0.75 | 87.5 | 39.6 | 63.6 |
| BaRP (Ours)  w_c=1.00 | 53.7 | 34.4 | 44.1 |
