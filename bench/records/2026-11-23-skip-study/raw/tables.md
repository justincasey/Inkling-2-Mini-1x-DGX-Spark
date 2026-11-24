### What the rule drops

On the A lane's routes of the prompts below (all 48 passes, every row); bytes from `intake/census.json`; windows: 120 random windows (seed 1) of 8 consecutive rows (1 x 8) and of four such streams (4 x 8), every stream starting at a random row of a random prompt, union rule.

| t | prompts | picks dropped | routing mass dropped | bytes dropped, one row | window bytes saved 1 x 8 | window bytes saved 4 x 8 |
|--:|---|--:|--:|--:|--:|--:|
| 0.15 | chat-capital, chat-sky | 7.7 % | 2.4 % | 7.2 % | 5.9 % | 5.3 % |
| 0.2 | chat-capital, chat-sky, raw-counting | 11.8 % | 4.6 % | 11.3 % | 10.8 % | 9.5 % |
| 0.3 | chat-capital, chat-sky, raw-counting | 26.2 % | 13.1 % | 25.4 % | 22.7 % | 19.4 % |

### Quality against the A lane

KL = KL(A || run) of the softmax over the 200,058 valid logits, per row, in nats (mean / p99 / max over the rows). `floor` = the A lane against the exact fixture on the same rows (what its own numerics already cost). Hidden = relative L2 of the whole plane per visit, mean over visits 1..48 (visit 0 is the embedding, equal), and its maximum; flips = (visit, row) pairs after visit 0 whose top-8 set differs from the A lane's.

| setting | prompt | rows | KL mean | KL p99 | KL max | top-1 equal | generated-row flips | hidden mean (max) | h_final | router flips |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| floor | chat-capital | 40 | 0.0025 | 0.0303 | 0.0416 | 40/40 | 0 | 0.041 (0.126) | 0.073 | 9.7 % |
| floor | chat-sky | 49 | 0.0024 | 0.0337 | 0.0416 | 49/49 | 0 | 0.025 (0.043) | 0.045 | 8.9 % |
| floor | raw-counting | 50 | 0.0002 | 0.0012 | 0.0015 | 50/50 | 0 | 0.066 (0.168) | 0.046 | 17.3 % |
| t0.15-renorm | chat-capital | 40 | 0.0822 | 0.7219 | 0.7913 | 37/40 | 0 | 0.104 (0.210) | 0.156 | 34.6 % |
| t0.15-renorm | chat-sky | 49 | 0.1256 | 1.3015 | 1.5935 | 41/49 | 0 | 0.099 (0.194) | 0.157 | 31.9 % |
| t0.2-renorm | chat-capital | 40 | 0.1695 | 1.1943 | 1.1975 | 35/40 | 0 | 0.172 (0.290) | 0.235 | 47.8 % |
| t0.2-renorm | chat-sky | 49 | 0.2769 | 2.5629 | 2.6896 | 37/49 | 1 | 0.167 (0.366) | 0.259 | 44.9 % |
| t0.2-renorm | raw-counting | 50 | 0.0036 | 0.0305 | 0.0326 | 50/50 | 0 | 0.153 (0.307) | 0.106 | 49.7 % |
| t0.3-renorm | chat-capital | 40 | 0.3318 | 2.3620 | 2.4170 | 31/40 | 1 | 0.267 (0.419) | 0.331 | 63.3 % |
| t0.3-renorm | chat-sky | 49 | 0.4322 | 2.6868 | 2.9057 | 36/49 | 0 | 0.272 (0.480) | 0.384 | 60.2 % |
| t0.3-renorm | raw-counting | 50 | 0.0231 | 0.1521 | 0.1670 | 50/50 | 0 | 0.290 (0.557) | 0.249 | 72.8 % |
| t0.2-plain | chat-capital | 40 | 0.0957 | 1.1795 | 1.2592 | 37/40 | 0 | 0.094 (0.160) | 0.158 | 34.6 % |
| t0.2-plain | chat-sky | 49 | 0.0969 | 1.0171 | 1.0548 | 42/49 | 1 | 0.090 (0.282) | 0.184 | 30.9 % |
| t0.2-plain | raw-counting | 50 | 0.0016 | 0.0066 | 0.0077 | 50/50 | 0 | 0.134 (0.268) | 0.084 | 45.7 % |
| d0.2-renorm | chat-capital | 15 | 0.0043 | 0.0369 | 0.0416 | 14/15 | 1 | 0.121 (0.307) | 0.113 | 37.3 % |
| d0.2-renorm | chat-sky | 23 | 0.0020 | 0.0118 | 0.0124 | 22/23 | 1 | 0.139 (0.263) | 0.178 | 40.3 % |
| d0.2-renorm | raw-counting | 15 | 0.0005 | 0.0017 | 0.0017 | 15/15 | 0 | 0.138 (0.287) | 0.111 | 41.0 % |
| d0.3-renorm | chat-capital | 15 | 0.0173 | 0.1101 | 0.1189 | 14/15 | 1 | 0.230 (0.324) | 0.191 | 54.3 % |
| d0.3-renorm | chat-sky | 23 | 0.0111 | 0.0571 | 0.0581 | 23/23 | 0 | 0.245 (0.389) | 0.280 | 54.6 % |

Pooled over the prompts a setting ran on (rows weighted equally):

| setting | prompts | rows | KL mean | KL p99 | KL max | KL mean, generated rows | floor KL mean (same rows) | KL mean / floor | top-1 equal |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| t0.15-renorm | chat-capital, chat-sky | 89 | 0.1061 | 1.0582 | 1.5935 | 0.0012 | 0.0024 | 44 x | 78/89 |
| t0.2-renorm | chat-capital, chat-sky, raw-counting | 139 | 0.1477 | 2.3151 | 2.6896 | 0.0030 | 0.0016 | 92 x | 122/139 |
| t0.3-renorm | chat-capital, chat-sky, raw-counting | 139 | 0.2562 | 2.4373 | 2.9057 | 0.0085 | 0.0016 | 159 x | 117/139 |
| t0.2-plain | chat-capital, chat-sky, raw-counting | 139 | 0.0623 | 1.0548 | 1.2592 | 0.0019 | 0.0016 | 39 x | 129/139 |
| d0.2-renorm | chat-capital, chat-sky, raw-counting | 53 | 0.0022 | 0.0264 | 0.0416 | 0.0022 | 0.0005 | 4 x | 51/53 |
| d0.3-renorm | chat-capital, chat-sky | 38 | 0.0135 | 0.0964 | 0.1189 | 0.0135 | 0.0007 | 20 x | 37/38 |

### Argmax flips against the A lane

Row t predicts token t + 1; `generated` rows are the ones whose argmax is a token the A lane greedily produced. Margin = the A lane's top-1 minus top-2 logit at the row; gap = the A lane's top-1 minus its logit at the run's pick.

| setting | prompt | flips | rows (generated?: margin / gap) |
|---|---|--:|---|
| t0.15-renorm | chat-capital | 3 | 5: 0.17 / 1.36; 6: 2.00 / 2.00; 7: 2.12 / 2.12 |
| t0.15-renorm | chat-sky | 8 | 5: 0.17 / 1.36; 6: 2.00 / 2.00; 7: 2.12 / 2.12; 12: 2.50 / 2.50; 15: 0.01 / 0.01; 16: 1.56 / 1.56; 17: 0.09 / 0.24; 18: 0.10 / 0.10 |
| t0.2-renorm | chat-capital | 5 | 5: 0.17 / 1.82; 6: 2.00 / 2.00; 10: 0.81 / 0.81; 14: 0.38 / 0.38; 23: 1.16 / 1.16 |
| t0.2-renorm | chat-sky | 12 | 5: 0.17 / 1.82; 6: 2.00 / 2.00; 12: 2.50 / 2.50; 14: 1.82 / 1.82; 15: 0.01 / 0.01; 16: 1.56 / 1.56; 17: 0.09 / 0.09; 18: 0.10 / 0.10; 22: 0.54 / 0.54; 23: 0.22 / 0.30; 24: 1.90 / 9.47; 30g: 0.03 / 0.03 |
| t0.2-renorm | raw-counting | 0 |  |
| t0.3-renorm | chat-capital | 9 | 3: 0.46 / 0.46; 5: 0.17 / 1.82; 6: 2.00 / 2.00; 8: 5.19 / 5.19; 10: 0.81 / 0.81; 16: 0.32 / 0.32; 19: 1.25 / 1.88; 23: 1.16 / 1.16; 35g: 0.04 / 0.04 |
| t0.3-renorm | chat-sky | 13 | 3: 0.46 / 0.46; 5: 0.17 / 1.82; 6: 2.00 / 2.00; 8: 5.19 / 5.19; 9: 1.24 / 1.56; 13: 1.39 / 1.39; 14: 1.82 / 1.82; 15: 0.01 / 0.01; 16: 1.56 / 1.56; 17: 0.09 / 0.09; 22: 0.54 / 0.54; 23: 0.22 / 0.30; 24: 1.90 / 9.47 |
| t0.3-renorm | raw-counting | 0 |  |
| t0.2-plain | chat-capital | 3 | 5: 0.17 / 1.82; 6: 2.00 / 2.00; 10: 0.81 / 0.81 |
| t0.2-plain | chat-sky | 7 | 5: 0.17 / 1.82; 6: 2.00 / 2.00; 12: 2.50 / 2.50; 15: 0.01 / 0.01; 16: 1.56 / 1.56; 17: 0.09 / 0.09; 30g: 0.03 / 0.03 |
| t0.2-plain | raw-counting | 0 |  |
| d0.2-renorm | chat-capital | 1 | 29g: 0.39 / 0.39 |
| d0.2-renorm | chat-sky | 1 | 30g: 0.03 / 0.03 |
| d0.2-renorm | raw-counting | 0 |  |
| d0.3-renorm | chat-capital | 1 | 29g: 0.39 / 0.39 |
| d0.3-renorm | chat-sky | 0 |  |
| floor | chat-capital | 0 |  |
| floor | chat-sky | 0 |  |
| floor | raw-counting | 0 |  |

### Hidden plane per visit (relative L2 against the A lane's plane; floor = A lane against exact)

**chat-capital**

| setting | v1 | v2 | v3 | v6 | v12 | v24 | v36 | v42 | v48 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| floor | 0.0027 | 0.0101 | 0.0113 | 0.0099 | 0.0186 | 0.0274 | 0.0682 | 0.0799 | 0.1167 |
| t0.15-renorm | 0.0544 | 0.0521 | 0.0908 | 0.0871 | 0.0910 | 0.1098 | 0.1118 | 0.1278 | 0.2100 |
| t0.2-renorm | 0.1679 | 0.1134 | 0.2471 | 0.1493 | 0.1396 | 0.2059 | 0.1713 | 0.2124 | 0.2902 |
| t0.3-renorm | 0.3703 | 0.2331 | 0.3473 | 0.2350 | 0.2646 | 0.3179 | 0.3297 | 0.2643 | 0.4186 |
| t0.2-plain | 0.0739 | 0.0616 | 0.0596 | 0.0612 | 0.0753 | 0.0891 | 0.0922 | 0.1097 | 0.1605 |
| d0.2-renorm | 0.0953 | 0.1008 | 0.1000 | 0.1209 | 0.1243 | 0.1409 | 0.1166 | 0.0865 | 0.3066 |
| d0.3-renorm | 0.2824 | 0.1864 | 0.2517 | 0.2039 | 0.2124 | 0.2752 | 0.3055 | 0.1740 | 0.3243 |

**chat-sky**

| setting | v1 | v2 | v3 | v6 | v12 | v24 | v36 | v42 | v48 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| floor | 0.0025 | 0.0066 | 0.0052 | 0.0057 | 0.0244 | 0.0258 | 0.0269 | 0.0324 | 0.0429 |
| t0.15-renorm | 0.0415 | 0.0556 | 0.0869 | 0.0787 | 0.1037 | 0.1156 | 0.0873 | 0.1154 | 0.1935 |
| t0.2-renorm | 0.1471 | 0.1196 | 0.2340 | 0.1395 | 0.1500 | 0.2103 | 0.1485 | 0.2000 | 0.3254 |
| t0.3-renorm | 0.3521 | 0.2438 | 0.3235 | 0.2193 | 0.2791 | 0.3271 | 0.3137 | 0.3233 | 0.4445 |
| t0.2-plain | 0.0634 | 0.0634 | 0.0565 | 0.0513 | 0.0778 | 0.0959 | 0.0779 | 0.0992 | 0.2425 |
| d0.2-renorm | 0.0816 | 0.1209 | 0.1232 | 0.1130 | 0.1530 | 0.1637 | 0.1331 | 0.1615 | 0.2102 |
| d0.3-renorm | 0.2213 | 0.2277 | 0.2228 | 0.1967 | 0.2571 | 0.3172 | 0.2452 | 0.2588 | 0.3388 |

**raw-counting**

| setting | v1 | v2 | v3 | v6 | v12 | v24 | v36 | v42 | v48 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| floor | 0.0079 | 0.0055 | 0.0111 | 0.0089 | 0.0555 | 0.1398 | 0.1681 | 0.1132 | 0.0317 |
| t0.2-renorm | 0.1805 | 0.1012 | 0.0979 | 0.0806 | 0.2105 | 0.2410 | 0.2356 | 0.1617 | 0.0728 |
| t0.3-renorm | 0.3844 | 0.2256 | 0.1874 | 0.1769 | 0.4016 | 0.4018 | 0.4175 | 0.2841 | 0.1710 |
| t0.2-plain | 0.1160 | 0.0657 | 0.0656 | 0.0677 | 0.1486 | 0.2384 | 0.2294 | 0.1568 | 0.0584 |
| d0.2-renorm | 0.1729 | 0.0866 | 0.0634 | 0.0641 | 0.1292 | 0.2155 | 0.2454 | 0.1885 | 0.0711 |

