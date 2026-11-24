# fp8-numerics: what an e4m3 prefill GEMM adds to the Mini's numerics (gate `fp8-numerics`, plan row 7.3 step 1)

Result **fail** (`meta.json`): the decision gate's condition (b) does not hold. "Fail" says the gate's condition is not met; the measurements
are complete and reproducible. **The fp8-added deviation is larger than the A lane's own, at every level measured: 60-181 times at the op, 1.5 to 2.6
times in the whole-forward hidden plane (the model amplifies both to the same order), and 4 to 10 times in the per-token KL on the chat prompts.**
The B lane is not worth building on numerics grounds alone; step 2 (`2026-11-23-fp8-potential`) says the same for speed.

## What was emulated

The EXL3 layer is `y = ((((x . suh) H_K) W_q) H_N) . svh`. The A lane rounds the rotated input `xh = (x . suh) H_K` to fp16 and multiplies by the
decoded codebook values `W_q` (fp16, exact) with fp32 accumulation. The fp8 emulation (`oracle/fp8_emu.py`, torch 2.14 `float8_e4m3fn`, round to
nearest even, saturating at 448, which is `cvt.rn.satfinite.e4m3x2`) replaces the two operands; float64 stands for fp32 accumulation, as the oracle's
own `moe_fp16` / `dense_fp16` do:

| variant | A operand (rotated input) | B operand (`W_q`) |
|---|---|---|
| `exact` | float64 | float64 |
| `A` | fp16 (the A lane, `Emulation.dense_fp16` / `moe_fp16`) | fp16 values, exact |
| `f8_row` | e4m3, one fp32 scale a row (amax / 448), applied in the epilogue | e4m3 of the fp16 value, no scale |
| `f8_blk128` | e4m3, one fp32 scale per (row, 128 columns of K) | as `f8_row` |
| `f8_row_tile` | as `f8_row` | e4m3 with one fp32 scale per 128 x 128 tile (amax / 448) |
| `f8_act_row` | as `f8_row` | fp16 values (attribution: activations only) |
| `f8_wt` | float64 (no activation rounding) | as `f8_row` (attribution: weights only) |

Everything else keeps the A lane's numerics (`Emulation.engine()`: int8 KV, bf16 P, the glue, the router, the controllers, the unembed); only the
EXL3 projections of attention (q, r, gate, k, v, o) and the routed experts' gate / up / down change. `A` run through the same harness equals the engine
fixture bit for bit on all three prompts (`equals_engine_fixture_bitwise`: true), so the harness changes nothing but the variant.

## Part 1: one op on real inputs

`fp8_emu.py layers` runs the real forward of a fixture prompt in the A lane's numerics and, at every dense EXL3 call (all 48 visits) and at the MoE of
visits 1, 8, 28, 40, 46 (layers 1, 8, 16, 16, 22), evaluates every variant on the REAL input rows of that call (the post-norm, post-mix rows, not a
synthetic distribution) with the real decoded tensors of the pack, and compares each with the exact float64 product. `moe_down_iso`: the down GEMM
fed the exact bf16 activation (its own error); `moe_expert_e2e`: gate, up, activation, down (errors compound); `moe_layer`: the weighted sum of the
experts (pre-bf16). Relative L2 over all rows of all calls; the A lane column is the number to compare with (its max-abs over the row rms, 3e-4 to
2.4e-3 here, is the record `kernels-dense`'s "5.6e-4 to 9.2e-4").

**raw-long-window (1,119 tokens)** (relative L2 against the float64 exact product; rows = 1119 prompt rows)

| kind | rows | A | f8_row | f8_blk128 | f8_row_tile | f8_act_row | f8_wt | f8_row / A |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| attn_q | 53712 | 1.78e-04 | 3.21e-02 | 3.16e-02 | 3.21e-02 | 2.27e-02 | 2.27e-02 | 180x |
| attn_r | 53712 | 1.46e-04 | 2.61e-02 | 2.59e-02 | 2.63e-02 | 1.86e-02 | 1.83e-02 | 179x |
| attn_gate | 53712 | 8.86e-05 | 1.61e-02 | 1.58e-02 | 1.60e-02 | 1.13e-02 | 1.13e-02 | 181x |
| attn_k | 53712 | 1.63e-04 | 2.95e-02 | 2.91e-02 | 2.95e-02 | 2.08e-02 | 2.09e-02 | 181x |
| attn_v | 53712 | 1.56e-04 | 2.82e-02 | 2.78e-02 | 2.83e-02 | 2.00e-02 | 1.99e-02 | 181x |
| attn_o | 53712 | 2.38e-04 | 4.29e-02 | 4.24e-02 | 4.29e-02 | 3.03e-02 | 3.04e-02 | 180x |
| moe_gate | 44760 | 1.54e-04 | 2.79e-02 | 2.75e-02 | 2.79e-02 | 1.97e-02 | 1.97e-02 | 181x |
| moe_up | 44760 | 1.82e-04 | 3.29e-02 | 3.25e-02 | 3.29e-02 | 2.33e-02 | 2.33e-02 | 181x |
| moe_down_iso | 44760 | 2.00e-04 | 3.61e-02 | 3.58e-02 | 3.61e-02 | 2.55e-02 | 2.56e-02 | 181x |
| moe_expert_e2e | 44760 | 9.08e-04 | 5.51e-02 | 5.45e-02 | 5.51e-02 | 3.89e-02 | 3.91e-02 | 61x |
| moe_layer | 5595 | 8.01e-04 | 4.79e-02 | 4.73e-02 | 4.79e-02 | 3.40e-02 | 3.38e-02 | 60x |

**chat-capital-effort0 (25 tokens)** (relative L2 against the float64 exact product; rows = 25 prompt rows)

| kind | rows | A | f8_row | f8_blk128 | f8_row_tile | f8_act_row | f8_wt | f8_row / A |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| attn_q | 1200 | 1.79e-04 | 3.23e-02 | 3.19e-02 | 3.23e-02 | 2.28e-02 | 2.28e-02 | 181x |
| attn_r | 1200 | 1.45e-04 | 2.62e-02 | 2.60e-02 | 2.63e-02 | 1.86e-02 | 1.85e-02 | 181x |
| attn_gate | 1200 | 8.34e-05 | 1.51e-02 | 1.49e-02 | 1.51e-02 | 1.06e-02 | 1.07e-02 | 181x |
| attn_k | 1200 | 1.41e-04 | 2.55e-02 | 2.52e-02 | 2.56e-02 | 1.81e-02 | 1.80e-02 | 181x |
| attn_v | 1200 | 1.35e-04 | 2.43e-02 | 2.40e-02 | 2.44e-02 | 1.72e-02 | 1.72e-02 | 181x |
| attn_o | 1200 | 2.30e-04 | 4.14e-02 | 4.09e-02 | 4.14e-02 | 2.93e-02 | 2.93e-02 | 180x |
| moe_gate | 1000 | 1.55e-04 | 2.81e-02 | 2.76e-02 | 2.80e-02 | 1.98e-02 | 1.99e-02 | 181x |
| moe_up | 1000 | 1.81e-04 | 3.27e-02 | 3.23e-02 | 3.27e-02 | 2.31e-02 | 2.32e-02 | 181x |
| moe_down_iso | 1000 | 1.97e-04 | 3.56e-02 | 3.52e-02 | 3.56e-02 | 2.52e-02 | 2.52e-02 | 181x |
| moe_expert_e2e | 1000 | 8.45e-04 | 5.21e-02 | 5.16e-02 | 5.23e-02 | 3.70e-02 | 3.68e-02 | 62x |
| moe_layer | 125 | 5.96e-04 | 4.04e-02 | 4.00e-02 | 4.06e-02 | 2.88e-02 | 2.85e-02 | 68x |

**Row distribution, `raw-long-window`** (per-row relative L2: median / p99 / max; max-abs over the row rms: mean / p99 / max)

| kind | variant | median | p99 | max | maxabs/rms mean | p99 | max |
|---|---|---:|---:|---:|---:|---:|---:|
| attn_q | A | 1.78e-04 | 2.04e-04 | 2.28e-04 | 6.62e-04 | 8.53e-04 | 1.13e-03 |
| attn_q | f8_row | 3.20e-02 | 3.64e-02 | 3.99e-02 | 1.20e-01 | 1.54e-01 | 1.98e-01 |
| attn_r | A | 1.53e-04 | 2.18e-04 | 2.60e-04 | 4.91e-04 | 7.63e-04 | 1.07e-03 |
| attn_r | f8_row | 2.75e-02 | 3.87e-02 | 4.61e-02 | 8.77e-02 | 1.36e-01 | 1.87e-01 |
| attn_gate | A | 8.88e-05 | 1.36e-04 | 1.97e-04 | 3.31e-04 | 5.14e-04 | 7.28e-04 |
| attn_gate | f8_row | 1.61e-02 | 2.37e-02 | 3.00e-02 | 6.09e-02 | 9.33e-02 | 1.28e-01 |
| attn_k | A | 1.62e-04 | 2.02e-04 | 2.35e-04 | 5.62e-04 | 7.65e-04 | 1.03e-03 |
| attn_k | f8_row | 2.94e-02 | 3.63e-02 | 3.96e-02 | 1.02e-01 | 1.39e-01 | 1.79e-01 |
| attn_v | A | 1.65e-04 | 2.26e-04 | 2.72e-04 | 5.67e-04 | 8.41e-04 | 1.19e-03 |
| attn_v | f8_row | 2.98e-02 | 4.10e-02 | 4.78e-02 | 1.03e-01 | 1.51e-01 | 2.19e-01 |
| attn_o | A | 2.43e-04 | 3.09e-04 | 3.49e-04 | 9.04e-04 | 1.34e-03 | 2.42e-03 |
| attn_o | f8_row | 4.38e-02 | 5.60e-02 | 6.03e-02 | 1.60e-01 | 2.26e-01 | 3.58e-01 |
| moe_gate | A | 1.63e-04 | 2.16e-04 | 2.36e-04 | 5.64e-04 | 8.09e-04 | 1.08e-03 |
| moe_gate | f8_row | 2.96e-02 | 3.89e-02 | 4.19e-02 | 1.02e-01 | 1.45e-01 | 2.01e-01 |
| moe_up | A | 1.85e-04 | 2.25e-04 | 2.48e-04 | 6.43e-04 | 8.68e-04 | 1.15e-03 |
| moe_up | f8_row | 3.36e-02 | 4.06e-02 | 4.35e-02 | 1.16e-01 | 1.57e-01 | 2.20e-01 |
| moe_down_iso | A | 2.01e-04 | 2.23e-04 | 2.45e-04 | 7.32e-04 | 9.34e-04 | 1.23e-03 |
| moe_down_iso | f8_row | 3.64e-02 | 3.93e-02 | 4.16e-02 | 1.32e-01 | 1.68e-01 | 2.28e-01 |
| moe_expert_e2e | A | 8.31e-04 | 1.99e-03 | 5.69e-03 | 3.29e-03 | 7.94e-03 | 2.35e-02 |
| moe_expert_e2e | f8_row | 5.63e-02 | 7.21e-02 | 8.34e-02 | 2.07e-01 | 2.91e-01 | 5.46e-01 |
| moe_layer | A | 7.23e-04 | 1.51e-03 | 2.72e-03 | 2.79e-03 | 5.85e-03 | 1.09e-02 |
| moe_layer | f8_row | 4.65e-02 | 6.57e-02 | 6.88e-02 | 1.72e-01 | 2.59e-01 | 3.26e-01 |

**Whole forward** (mean over visits 1..48 of the hidden plane's relative L2; the A lane = engine fixture)

| prompt | rows | floor: A lane vs exact (mean / max) | variant | added: run vs A lane (mean / max) | visits with added <= floor | run vs exact (mean) | argmax = exact | top-2 = exact | KL(exact‖run) mean / p99 |
|---|---:|---|---|---|---:|---:|---:|---:|---|
| chat-capital-effort0 | 40 | 0.0405 / 0.1256 | A lane (floor) | 0 (bitwise equal to the engine fixture: True) | - | 0.0405 | 40/40 | 36/40 | 0.0028 / 0.0365 |
|  |  |  | f8_row | 0.0841 / 0.2099 | 0/48 | 0.0804 | 39/40 | 32/40 | 0.0260 / 0.1871 |
|  |  |  | f8_blk128 | 0.0696 / 0.1966 | 0/48 | 0.0709 | 38/40 | 31/40 | 0.0300 / 0.2445 |
|  |  |  | f8_row_tile | 0.0619 / 0.0975 | 8/48 | 0.0691 | 40/40 | 36/40 | 0.0151 / 0.0980 |
|  |  |  | f8_act_row | 0.0592 / 0.1315 | 0/48 | 0.0525 | 37/40 | 31/40 | 0.0234 / 0.3612 |
|  |  |  | f8_wt | 0.0650 / 0.1152 | 2/48 | 0.0654 | 40/40 | 36/40 | 0.0134 / 0.1170 |
| chat-sky-effort0.9 | 49 | 0.0254 / 0.0433 | A lane (floor) | 0 (bitwise equal to the engine fixture: True) | - | 0.0254 | 49/49 | 45/49 | 0.0025 / 0.0364 |
|  |  |  | f8_row | 0.0653 / 0.1020 | 0/48 | 0.0644 | 43/49 | 38/49 | 0.0264 / 0.2306 |
|  |  |  | f8_blk128 | 0.0641 / 0.1062 | 0/48 | 0.0635 | 45/49 | 37/49 | 0.0472 / 0.5396 |
|  |  |  | f8_row_tile | 0.0653 / 0.1135 | 0/48 | 0.0644 | 46/49 | 42/49 | 0.0236 / 0.1849 |
|  |  |  | f8_act_row | 0.0517 / 0.0811 | 0/48 | 0.0531 | 46/49 | 42/49 | 0.0088 / 0.0831 |
|  |  |  | f8_wt | 0.0521 / 0.0847 | 0/48 | 0.0514 | 47/49 | 41/49 | 0.0113 / 0.0796 |
| raw-counting | 50 | 0.0663 / 0.1681 | A lane (floor) | 0 (bitwise equal to the engine fixture: True) | - | 0.0663 | 50/50 | 50/50 | 0.0002 / 0.0011 |
|  |  |  | f8_row | 0.1017 / 0.2028 | 0/48 | 0.0858 | 50/50 | 48/50 | 0.0007 / 0.0042 |
|  |  |  | f8_blk128 | 0.1040 / 0.2077 | 0/48 | 0.0921 | 50/50 | 49/50 | 0.0007 / 0.0035 |
|  |  |  | f8_row_tile | 0.1113 / 0.2205 | 0/48 | 0.1103 | 50/50 | 50/50 | 0.0007 / 0.0027 |
|  |  |  | f8_act_row | 0.0902 / 0.1926 | 3/48 | 0.0782 | 50/50 | 50/50 | 0.0004 / 0.0013 |
|  |  |  | f8_wt | 0.0917 / 0.1958 | 0/48 | 0.0829 | 50/50 | 49/50 | 0.0005 / 0.0036 |

**First visits, `f8_row`: added deviation / the A lane's floor** (visit 1, 2, 3, 4, 6, 12, 24, 48)

| prompt | 1 | 2 | 3 | 4 | 6 | 12 | 24 | 48 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chat-capital-effort0 | 0.035 / 0.0027 = 13.0x | 0.032 / 0.0101 = 3.1x | 0.029 / 0.0113 = 2.6x | 0.030 / 0.0101 = 3.0x | 0.028 / 0.0099 = 2.8x | 0.054 / 0.0186 = 2.9x | 0.080 / 0.0274 = 2.9x | 0.185 / 0.1167 = 1.6x |
| chat-sky-effort0.9 | 0.037 / 0.0025 = 14.4x | 0.033 / 0.0066 = 5.1x | 0.029 / 0.0052 = 5.4x | 0.028 / 0.0055 = 5.1x | 0.026 / 0.0057 = 4.6x | 0.054 / 0.0244 = 2.2x | 0.068 / 0.0258 = 2.6x | 0.091 / 0.0429 = 2.1x |
| raw-counting | 0.040 / 0.0079 = 5.0x | 0.037 / 0.0055 = 6.7x | 0.035 / 0.0111 = 3.1x | 0.033 / 0.0124 = 2.7x | 0.027 / 0.0089 = 3.1x | 0.111 / 0.0555 = 2.0x | 0.193 / 0.1398 = 1.4x | 0.046 / 0.0317 = 1.5x |

## Reading

* **Op level.** Every fp8 variant is 60 to 181 times the A lane's relative error against the exact product: 1.6-4.3% against 0.009-0.024% (dense), 2.8-3.6% against 0.015-0.020% (MoE gate / up / down). The
  per-row maximum (`row_rel_max`) is 3-6% against 0.03%; max-abs over the row rms is 0.06-0.36 against 0.0003-0.0024. An e4m3 operand has 3
  mantissa bits: a Gaussian fp16 codebook value loses 2.65% rms (measured: 0.0265 on N(0,1)); the two operands give 2.7% x sqrt(2) = 3.7%, which
  is what the MoE table shows (`f8_act_row` and `f8_wt` are each 70% of `f8_row`: the weights and the activations contribute equally).
* **The mitigations do not help.** A scale a (row, 128 columns) (`f8_blk128`) and a per-tile weight scale (`f8_row_tile`) change nothing (the rotated activations are
  Gaussian, there are no outliers to protect; scales move the exponent, not the 3 mantissa bits). The only lever left in a one-pass e4m3 GEMM is more mantissa bits, i.e. two fp8 terms (twice the mma: no gain over fp16).
* **Whole forward** (below). The A lane is already far from the exact oracle in the hidden plane (0.025-0.066 mean over the 48 visits, up to 0.17: the model amplifies
  fp noise, `docs/ORACLE-GAP.md`), and the fp8 lane adds 1.5 to 2.6 times that on top of the A lane, because both saturate at the same order of amplification. The first visits show the
  real ratio before saturation: 13x and 14x at visit 1 on the chat prompts, 3-6x for the next visits. At the logits the cost is visible: KL(exact || run) is 0.026 (`f8_row`) against 0.0028
  (A lane) on `chat-capital-effort0`, 0.026 against 0.0025 on `chat-sky-effort0.9`, 0.0007 against 0.0002 on `raw-counting`; the argmax differs from the exact oracle on 1 of 40 and 6 of 49
  chat rows (the A lane: 0 and 0), top-2 order on 8 of 40 and 11 of 49 (the A lane: 4 and 4).
* **Gate (b)** (the fp8-added deviation per visit no larger than the A lane's engine-vs-exact deviation): `f8_row` exceeds it at 48 of 48 visits on all three prompts (the smallest
  ratio 1.05 on `raw-counting`, 1.31 and 1.88 on the chat prompts); the best variant of the attribution set (weights only, `f8_wt`) at 46 of 48 on the first prompt and 48 of 48 on the others, so
  none passes. **(b) fails.**

## Not measured

* The whole forward of the 1,119-token prompt in fp8 (the op-level probe ran on it: Part 1, all 1,119 rows): the short prompts already decide the gate, and the fixtures of that prompt are 2.3 GB.
* fp8 on the attention core (QK, PV) was never proposed (the order of work leaves it to a long-context potential that step 2 does not show).
* Quality gates of step 3 (exact-answer set, tools, needle, teacher-forced KL on 50,000 tokens): not run; the gate stopped the lane before them.

## Reproduce

Node-4 CPU (`nice -n 19`, `OMP_NUM_THREADS=1`, 4-6 workers), the oracle's venv (torch 2.14.1+cpu, numpy 2.5.5), pack at the pinned revision:

```text
python oracle/fp8_emu.py layers  --pack PACK --fixture <fixture with the prompt's tokens> --out layers.json --workers 5 --threads 2       # raw/layers-*.json (long: 910 s, chat-capital: 126 s)
python oracle/fp8_emu.py forward --pack PACK --fixture <fixture> --variant {A,f8_row,f8_blk128,f8_row_tile,f8_wt,f8_act_row} --out fwd/<variant>-<prompt>.safetensors
python oracle/fp8_summary.py --spec fixtures/spec --engine fixtures/engine --runs fwd --prompts chat-capital-effort0 chat-sky-effort0.9 raw-counting --out summary.json   # raw/summary.json
```

`raw/` holds the two layer reports, the summary (per-visit curves for every run) and the logs; the fixtures are the record `oracle-fixtures`'s (CPU float64 runs, they stay on the node).
