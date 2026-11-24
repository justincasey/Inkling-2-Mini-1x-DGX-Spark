# Roofline and targets

Arithmetic on hardware ceilings and the pack census; no engine number is in this file (there is no engine for this
model yet: `PLAN.md`). Every table regenerates from `python3 scripts/roofline.py` (census + config, offline) and
`python3 scripts/official_bytes.py` (the official checkpoint's header, HTTP range). Targets are **proposals** until the
owner confirms them: the Small recipe marks a number the owner set with "(user)", and none below is.

## Hardware (one GB10)

| Fact | Value | Source |
|---|---|---|
| sustained read bandwidth | **233.8 GB/s** streaming read (8 GiB); REXL3's EXL3 kernels are profiled against **237 GB/s** ("DRAM read probe 233-237 GB/s is the hard ceiling") | GLM-5.3-EXL3-3.38bpw-4x `docs/ROOFLINE.md` (`tools/roofline.cu`, a GB10 node, 2026-11-04); CYBERFROST `docs/exec-plans/active/cyberfrost-rexl3.md:306` |
| higher figures seen | 258 GB/s "a plain stream reaches"; ~270 GB/s of trellis in a 4-bit EXL3 one-row linear | Inkling-2-Small `docs/exec-plans/completed/autoresearch-segment-a.md:226-227`; skill `gb10-cuda-kernel-perf` |
| dense bf16 / fp16 / fp8 e4m3, 8192³ | **93.4 / 91.9 / 195.1 TFLOP/s**; register-only mma peak 107.4 | GLM-5.3-EXL3-3.38bpw-4x `docs/ROOFLINE.md`; CYBERFROST exec plan `:371` |
| SMs / L2 / shared-memory opt-in | 48 / 24 MiB / 101,376 B | GLM-5.3-EXL3-3.38bpw-4x `docs/ROOFLINE.md` |
| memory | MemTotal 121.69 GiB; ~11 GiB OS + CUDA context; MemAvailable kept >= 10 GiB under load | Inkling-2-Small `docs/FIT.md`; CYBERFROST README |
| GPU clock | locked at 2,200 MHz (site policy), sampled 2,184-2,197 MHz under load | CYBERFROST exec plan `:122`, `:626` |
| NVMe | ~12 GB/s sequential, `O_DIRECT` | GLM-5.3-EXL3-3.38bpw-4x `docs/ROOFLINE.md` |

**Not re-measured for this recipe.** On 2026-11-22 all four Sparks held Inkling-2-Small TP4 ranks at 95-96% GPU
utilisation with 11.2-12.7 GiB of MemAvailable (read-only probe: `nvidia-smi`, `/proc/meminfo`, `ps`), so
`tools/roofline.cu` was not run: it would have measured the other workload and risked the ring. Before the first boot,
on the node that will serve this recipe:

```bash
nvcc -O3 -arch=sm_121 tools/roofline.cu -o roofline -lcublasLt -lcublas && ./roofline   # prints METRIC lines
```

If `read_gbs` or the bf16 figure differs from the table by more than a few percent, update `HW` in
`scripts/roofline.py` and regenerate. The last two GEMM shapes in `roofline.cu` are the GLM recipe's; this model's
prefill shapes (4096 x 2304 x 3072, and 256-row expert GEMMs of 2304 x 1152) are not in the tool yet.

## Pack (census, `intake/census.json`)

A token makes **48 layer passes**: 24 layers, the middle 12 (6-17) run three times with shared weights. 8 of the passes
are global attention (layers 5, 11 x3, 17 x3, 23) and 40 are sliding-window (1,024). Decode bytes follow the passes,
not the 14 GiB on disk: every looped layer is read three times.

| Class | GiB | Format |
|---|---:|---|
| routed experts (24 x 128 x gate/up/down) | 11.843 | EXL3 4-bit (7,944 tensors) and 5-bit (1,272 = 13.8%) |
| attention (q, k, v, o, gate, rel-logits per layer) | 0.457 | EXL3 6-bit |
| embedding | 0.863 | BF16 (one row read per token) |
| unembed | 0.324 | EXL3 6-bit |
| vision tower | 0.226 | EXL3 6-bit |
| loop controllers (12 looped layers) | 0.053 | BF16 |
| drafter (`mtp.safetensors`, 3 layers + Markov head) | 0.220 | EXL3 4-bit + BF16 |
| **resident** | **14.000** | matches the card's 14.00 GiB |

Mean K over all trellis tensors is 4.238 (the card's 4.23). SAGE put the 5-bit tensors where the loop amplifies error:
none in layers 0-5, up to 61% of layer 17's expert tensors, and 22.8% of `down_proj` against 9.3% of gate/up. A mean
expert is 4.00 MB in layers 0-5 and 4.61 MB in layer 17.

## Decode

Bytes one row reads, uniform routing (8 of 128 experts per pass):

| Item | GB |
|---|---:|
| routed experts | 1.62 |
| attention | 0.98 |
| unembed | 0.35 |
| loop controllers | 0.17 |
| routers | 0.03 |
| **weights, one row** | **3.14** |

Serial ceiling **75.4 tok/s** at 237 GB/s (13.26 ms); 74.4 at 233.8; 82.1 at 258. Routing is assumed uniform: if every
pick landed on the heaviest 8 experts of its layer the row would be 3.33 GB (+6%, 71.2 tok/s), the bound on that
assumption. The official checkpoint (BF16 attention and unembed, NVFP4 experts) reads **5.55 GB** a row: 42.7 tok/s at
the same bandwidth. This pack's decode ceiling is 1.77x higher.

### Context, one stream, serial (KV read added, 237 GB/s)

Sliding passes read a bf16 ring of 1,031 positions (0.17 GB a step).

| context | KV bf16 | KV int8 | KV q4 |
|---:|---:|---:|---:|
| 4K | 68.8 | 70.1 | 70.8 |
| 32K | 54.0 | 61.5 | 65.9 |
| 128K | 31.2 | 43.3 | 53.2 |
| 256K | 19.9 | 31.0 | 42.4 |
| 1M | 6.3 | 11.5 | 19.1 |

Long-context decode is a KV-bandwidth problem: at 256K an int8 step reads 4.3 GB of KV against 3.1 GB of weights.

### A window of rows shares one weight read, but not the experts

Rows in a verify window (or streams in a batch) pick experts independently, so the union grows fast: 128 x (1 - (15/16)^n).

| rows | experts / layer | GB | ms at 237 | x a serial step's bytes |
|---:|---:|---:|---:|---:|
| 1 | 8.0 | 3.14 | 13.3 | 1.00 |
| 2 | 15.5 | 4.66 | 19.7 | 1.48 |
| 3 | 22.5 | 6.08 | 25.6 | 1.93 |
| 4 | 29.1 | 7.41 | 31.3 | 2.36 |
| 6 | 41.1 | 9.83 | 41.5 | 3.13 |
| 8 | 51.6 | 11.95 | 50.4 | 3.80 |
| 16 | 82.4 | 18.17 | 76.7 | 5.78 |

This model is expert-bound (52% of a row), unlike CYBERFROST (512 experts, top 10, a large dense share), so drafting
buys less here. The drafter costs 1.0 ms a round for its weights; if its Markov head re-reads a folded `[201024, 256]`
bf16 table on each of 7 steps (Inkling-2-Small `docs/PORT-SPEC.md` 4.4) it costs 4.0 ms `[INFERENCE]`.

MTP, tokens per second = T / round, round = window + drafter, engine factor 1.08, 0.3 ms idle (drafter at its high bound):

| rows | round ms | tok/s at T = 2.0 / 2.5 / 3.0 / 3.5 | T for 75 tok/s | T for 100 tok/s |
|---:|---:|---|---:|---:|
| 2 | 25.9 | 77 / 97 / 116 / 135 | 1.9 | 2.6 |
| 3 | 32.4 | 62 / 77 / 93 / 108 | 2.4 | 3.2 |
| 4 | 38.4 | 52 / 65 / 78 / 91 | 2.9 | 3.8 |
| 6 | 49.4 | 40 / 51 / 61 / 71 | 3.7 | 4.9 |
| 8 | 59.1 | 34 / 42 / 51 / 59 | 4.4 | 5.9 |

T and the expert overlap between a window's rows are **not measured on this model**. Two scenarios bracket them:

- **conservative**: independent picks, drafter at its high bound, flat per-position acceptance 0.69 (Inkling-2-Small's
  DSpark: T = 3.04 at depth 7, `bench/records/2026-11-22-segment-a-v1/harness-runs/run20-a73ff7a-final/metrics.txt`).
- **sibling-calibrated**: distinct-expert counts 0.87 / 0.78 / 0.73 / 0.69 of independent at 2 / 4 / 6 / 8 rows
  (measured on GLM-5.3-Flash, GLM-5.3-EXL3 `docs/ROOFLINE.md`), drafter at its low bound, CYBERFROST's measured
  acceptance 73.6 / 77 / 82 / 89 / 86 / 79% (exec plan `:187`).

| rows | T flat / CF | conservative | sibling-calibrated | ceiling (f = 1.00) |
|---:|---|---:|---:|---:|
| 2 | 1.69 / 1.74 | 65 | 83 | 91 |
| 3 | 2.17 / 2.30 | 67 | 91 | 99 |
| 4 | 2.49 / 2.77 | 65 | 95 | 104 |
| 6 | 2.88 / 3.54 | 58 | 99 | 108 |
| 8 | 3.06 / 4.04 | 52 | 98 | 107 |

Serial is 69.8 tok/s at f = 1.08. **Under the conservative scenario drafting does not beat serial at c = 1**; under the
sibling scenario it gains 30-40%. Which one holds is the first thing to measure on the engine (route log + per-position
acceptance), before any depth is chosen.

At depth the KV read is paid once per round, so drafting gains more (4-row window, int8 KV):

| context | serial f=1.00 | serial f=1.08 | MTP conservative | MTP sibling-calibrated |
|---:|---:|---:|---:|---:|
| 4K | 70.1 | 64.9 | 63 | 92 |
| 32K | 61.5 | 56.9 | 60 | 85 |
| 128K | 43.3 | 40.1 | 51 | 70 |
| 256K | 31.0 | 28.7 | 42 | 56 |
| 1M | 11.5 | 10.6 | 21 | 25 |

### Concurrent streams (no drafting; one row per stream, int8 KV at 4K each)

| streams | GB / step | aggregate tok/s f=1.00 | f=1.08 | per stream f=1.08 |
|---:|---:|---:|---:|---:|
| 1 | 3.38 | 70 | 65 | 64.9 |
| 2 | 5.13 | 92 | 86 | 42.8 |
| 4 | 8.35 | 113 | 105 | 26.3 |
| 8 | 13.84 | 137 | 127 | 15.9 |
| 16 | 21.96 | 173 | 160 | 10.0 |

## Prefill

FLOPs per prompt token: 8.90 G of linears (2 x the weights one token multiplies over 48 passes, 8 of 128 experts) +
0.50 G of sliding attention + 49,152 x L of global attention (8 passes x 4 x 3,072 x L/2).

| prompt | GFLOP / token | bf16 ceiling (93.4) | 25% of it | 40% of it |
|---:|---:|---:|---:|---:|
| 8K | 9.81 | 9,525 | 2,381 | 3,810 |
| 32K | 11.01 | 8,481 | 2,120 | 3,392 |
| 128K | 15.85 | 5,894 | 1,474 | 2,358 |
| 256K | 22.29 | 4,191 | 1,048 | 1,676 |
| 512K | 35.17 | 2,655 | 664 | 1,062 |
| 1M | 60.94 | 1,533 | 383 | 613 |

A 4,096-row chunk reads 27.4 GB (every expert, every pass), 116 ms at 237 GB/s = 35K tok/s: bandwidth is not the limit.
Tensor work, trellis decode and launch structure are. Full attention equals the linears' cost at L = 180K.
GLM's engine showed the kernels are bound by trellis decode, latency and sync, not the tensor pipe
(GLM-5.3-EXL3 `docs/ROOFLINE.md`), so fp8 math alone does not lift the ceiling; treat 40% as a stretch.

## Capacity (one Spark)

| KV | bytes / token | + drafter plane | one 1,048,576-token lane |
|---|---:|---:|---:|
| bf16 | 32,768 | 36,864 | 36.00 GiB |
| int8 + per-(token, head) exponent | 16,512 | 18,576 | 18.14 GiB |
| q4 (the card) | 8,704 | 9,792 | 9.56 GiB |

The card's 8,704 / 9,792 B/token are reproduced exactly by 8 global passes and one drafter plane. Sliding rings are
0.157 GiB a slot.

Budget: 121.69 MemTotal - 11.0 OS/CUDA - 10.0 MemAvailable floor = 100.7 GiB; - 14.00 weights - 5.0 activations and
graphs = **81.7 GiB** for KV, slot state and the prefix cache.

| prefix cache | slots | KV GiB | int8 | bf16 |
|---:|---:|---:|---|---|
| 0 | 1 | 81.5 | 4.71M tokens (4.5 lanes of 1M) | 2.37M (2.3) |
| 8 GiB | 4 | 73.1 | 4.22M (4.0) | 2.13M (2.0) |
| 12 GiB | 4 | 69.1 | 3.99M (3.8) | 2.01M (1.9) |

Four 1M int8 lanes fit with 0.5 GiB to spare: no margin for the 10 GiB floor to move. The engine family this recipe
starts from keeps int8 KV only (`inkling2` accepts nothing else), so bf16 is a fallback, not a plan.

## Sibling calibration (what REXL3 engines reached against this arithmetic)

| Sibling | Result | Source |
|---|---|---|
| CYBERFROST, 1 Spark, final engine | decode window ~8% above its 237 GB/s byte floor; EXL3 linears and lm_head at 94-100%, MoE ~96% | exec plan `:459-462` |
| Inkling-2-Small, TP4, run 20 | decode round 56.2 ms against a 53 ms floor at 258 GB/s ("within about 6%"); ~15% above a 237 GB/s floor `[INFERENCE]` | `autoresearch-segment-a.md:266-267` |
| Inkling-2-Small, run 20, prefill | 2,161 / 1,868 / 1,575 tok/s at 32K / 128K / 256K = 22.6% / 24.5% / 26.0% of 93.4 TFLOP/s per GPU `[INFERENCE: ROOFLINE.md's 39 / 49 / 62 GFLOP a token over 4 GPUs]` | run20 `metrics.txt`, `docs/ROOFLINE.md` |
| CYBERFROST, first full run vs final | decode 49 -> 78.4, prefill 280 -> 1,996 tok/s: a first boot lands near 60% of final decode and 15% of final prefill | README Benchmarks |

So a mature engine runs decode at f = 1.08-1.15 and prefill at 23-26% of bf16 peak. A first boot far below that (serial
under ~40 tok/s) points at a bug, not a tuning gap `[INFERENCE]`.

## Targets (proposals, single Spark, GPU locked at 2,200 MHz, nothing else on the node)

Rules: serial = 95% of the f = 1.08 figure; MTP = midpoint of the conservative and sibling-calibrated scenarios until
the engine measures T and overlap; concurrency = 85% of the f = 1.08 aggregate; prefill = 25% of the bf16 ceiling
(the siblings' 23-26%), stretch 40%.

| Metric | Ceiling | Target | Stretch |
|---|---:|---:|---:|
| decode c=1, 4K, serial | 70.1 | **62** | 65 |
| decode c=1, 4K, MTP | 108 | **77** | 92 |
| decode c=1, 32K / 128K / 256K / 1M, serial (int8 KV) | 61.5 / 43.3 / 31.0 / 11.5 | 54 / 38 / 27 / 10 | 57 / 40 / 29 / 11 |
| decode c=1, 32K / 128K / 256K / 1M, MTP (4-row window) | | 72 / 60 / 49 / 23 | 85 / 70 / 56 / 25 |
| decode aggregate, 4 / 8 streams | 113 / 137 | 89 / 108 | 105 / 127 |
| prefill cold, 8K / 32K | 9,525 / 8,481 | 2,380 / 2,120 | 3,810 / 3,390 |
| prefill cold, 128K / 256K / 1M | 5,894 / 4,191 / 1,533 | 1,470 / 1,050 / 380 | 2,360 / 1,680 / 610 |
| ready after `start`, cold page cache | | <= 20 s | |

Boot: 14 GiB at ~12 GB/s is 1.3 s of disk; CYBERFROST needs 26.9 s for 52 GiB and RED-SNOW 70-85 s, so loader and
warm-up dominate. 20 s is a guess scaled from those, not an arithmetic bound `[INFERENCE]`.

## What moves the ceiling

1. **Bytes.** Attention is 31% of a token at 6 bits, the unembed 11%, BF16 controllers 5%. Attention at 5 or 4 bits
   would cut 5-10% of the bytes (+5-12% decode). It changes the pack, so it has to happen before a byte-exact baseline
   is pinned (the Small recipe's note on its BF16 attention).
2. **Window policy.** Experts are 52% of a row, so the window length and a confidence cut matter more than in
   CYBERFROST (its scored-budget policy averages 2.8 drafted rows).
3. **int8 sliding rings** (0.17 GB, 5% of a step at 4K) and **q4 KV** (256K serial ceiling 31.0 -> 42.4).
4. **Prefill structure:** decode-once tiles, bigger chunks (8,192), overlap of dequant and GEMM (skill
   `gb10-cuda-kernel-perf`).

## Unknown until the engine boots

- Expert overlap between consecutive tokens and across loop passes (a route log), and the drafter's per-position
  acceptance: they decide whether MTP is worth enabling at all.
- Route frequency of the 5-bit experts (bounded above: +6% bytes).
- `roofline.cu` on the serving node (above) and the clock under this model's load.
