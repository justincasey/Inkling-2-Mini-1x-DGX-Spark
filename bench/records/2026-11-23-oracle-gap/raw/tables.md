| prompt | rows | comparison | hidden rel L2 (worst visit) | h_final rel L2 | logits rel L2 plane / row median / row max | logits cos min | KL mean / max | argmax equal | top-2 equal | flips of pairs |
|---|---|---|---|---|---|---|---|---|---|---|
| chat-capital-effort0 | 40 | engine vs engine fixture | 0.11 (47) | 0.0774 | 0.0298 / 0.0206 / 0.176 | 0.98646 | 0.00734 / 0.0822 | 39/40 | 36/40 | 238 / 1920 (12.4%) |
|  |  | engine vs spec fixture | 0.0637 (40) | 0.062 | 0.0259 / 0.0188 / 0.198 | 0.98454 | 0.00691 / 0.0795 | 39/40 | 36/40 | 245 / 1920 (12.8%) |
|  |  | floor: engine fixture vs spec fixture | 0.126 (47) | 0.0728 | 0.0253 / 0.0108 / 0.0776 | 0.99863 | 0.00277 / 0.0512 | 40/40 | 36/40 | 182 / 1920 (9.5%) |
| chat-sky-effort0.9 | 49 | engine vs engine fixture | 0.0463 (48) | 0.0506 | 0.0191 / 0.0137 / 0.0864 | 0.99724 | 0.00411 / 0.0678 | 47/49 | 44/49 | 213 / 2352 (9.1%) |
|  |  | engine vs spec fixture | 0.0489 (37) | 0.0462 | 0.017 / 0.0148 / 0.0943 | 0.99599 | 0.00493 / 0.0654 | 47/49 | 46/49 | 229 / 2352 (9.7%) |
|  |  | floor: engine fixture vs spec fixture | 0.0433 (40) | 0.0451 | 0.0166 / 0.00972 / 0.0952 | 0.99589 | 0.00246 / 0.0512 | 49/49 | 45/49 | 204 / 2352 (8.7%) |
| raw-counting | 50 | engine vs engine fixture | 0.178 (24) | 0.0491 | 0.0247 / 0.016 / 0.172 | 0.98772 | 0.000283 / 0.00348 | 50/50 | 50/50 | 420 / 2400 (17.5%) |
|  |  | engine vs spec fixture | 0.159 (36) | 0.0475 | 0.0239 / 0.0173 / 0.0994 | 0.99505 | 0.000268 / 0.0028 | 50/50 | 50/50 | 463 / 2400 (19.3%) |
|  |  | floor: engine fixture vs spec fixture | 0.168 (36) | 0.0459 | 0.0253 / 0.014 / 0.188 | 0.98747 | 0.000166 / 0.00152 | 50/50 | 50/50 | 407 / 2400 (17.0%) |
| raw-long-window | 1126 | engine vs engine fixture | 0.256 (36) | 0.111 | 0.103 / 0.08 / 1.04 | 0.87126 | 0.000718 / 0.0923 | 1126/1126 | 1112/1126 | 14827 / 54048 (27.4%) |
|  |  | engine vs spec fixture | 0.26 (36) | 0.0829 | 0.0786 / 0.0582 / 0.56 | 0.90058 | 0.000231 / 0.012 | 1126/1126 | 1110/1126 | 14589 / 54048 (27.0%) |
|  |  | floor: engine fixture vs spec fixture | 0.267 (36) | 0.106 | 0.0897 / 0.0735 / 0.561 | 0.88274 | 0.000681 / 0.016 | 1126/1126 | 1115/1126 | 15414 / 54048 (28.5%) |

| prompt | engine / floor, per visit 1..48 (max at visit; min) | flips that recur in the floor (engine vs spec): share vs the floor's rate | smallest router margin | weights, max abs diff (identical sets) |
|---|---|---|---|---|
| chat-capital-effort0 | 2.19 (visit 8); 0.35 | 111/245 = 45.3% vs 9.5% | 1.31e-05 | 0.27 |
| chat-sky-effort0.9 | 1.29 (visit 13); 0.74 | 126/229 = 55.0% vs 8.7% | 3.13e-07 | 0.299 |
| raw-counting | 1.32 (visit 31); 0.58 | 289/463 = 62.4% vs 17.0% | 2.14e-08 | 0.552 |
| raw-long-window | 1.09 (visit 5); 0.76 | 11371/14589 = 77.9% vs 28.5% | 1.67e-07 | 1.12 |

| prompt | generated | engine tokens vs spec fixture's | vs engine fixture's | oracle top-1 - top-2 margin at the generated positions: min (position); positions under 0.3 |
|---|---|---|---|---|
| chat-capital-effort0 | 16 | equal | equal | 0.103 (11); 2 of 16 |
| chat-sky-effort0.9 | 24 | equal | equal | 0.0688 (5); 1 of 24 |
| raw-counting | 16 | equal | equal | 3.66 (8); 0 of 16 |
| raw-long-window | 8 | equal | equal | 2.41 (6); 0 of 8 |

| oracle margin of the row | [0, 0.05) | [0.05, 0.1) | [0.1, 0.2) | [0.2, 0.3) | [0.3, 0.5) | [0.5, 1) | [1, inf) |
|---|---|---|---|---|---|---|---|
| rows (spec fixture's margins, all prompts) | 0 | 3 | 7 | 2 | 6 | 5 | 1242 |
| argmax differs, engine vs engine fixture | 0 | 0 | 2 | 1 | 0 | 0 | 0 |
| argmax differs, engine vs spec fixture | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| argmax differs, floor: engine fixture vs spec fixture | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
