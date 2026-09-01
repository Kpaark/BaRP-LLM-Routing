# Mixture evaluation (OOD)

- **spec:** `ood_hard_gsm8k`  |  **BaRP w_c:** 0.00  |  **fallback:** most-expensive (gpt-4-1106-preview)  |  N = 7,450

| Method | GSM8K | Avg | Cost ($) |
| --- | --- | --- | --- |
| BaRP alone (w_c=0.00) | 54.8 | 54.8 | 0.00039 |
| Fallback alone: most-expensive (gpt-4-1106-preview) | 65.9 | 65.9 | 0.00855 |
| Mixture p=0.00 | 54.8 | 54.8 | 0.00039 |
| Mixture p=0.10 | 56.0 | 56.0 | 0.00124 |
| Mixture p=0.25 | 57.7 | 57.7 | 0.00248 |
| Mixture p=0.50 | 60.4 | 60.4 | 0.00449 |
| Mixture p=0.75 | 63.1 | 63.1 | 0.00650 |
| Mixture p=1.00 | 65.9 | 65.9 | 0.00855 |
