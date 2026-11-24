# prefill-push: cold prefill with the shipping configuration after the push for 3,000 tok/s (gate `speed-prefill`)

Result **pass** with the byte-identical proofs of `2026-11-23-prefill-push-exact` (same engine commit). Engine `ce650e376cdeb8eff6b4d63f6ae63facc95a0d85` (branch `inkling2-loop-prefill` on the integration head
`1d53344`: the drafter, the cache with images, the arena, the review fixes), a lab build, node-3 alone on the GPU, SM clock locked at 2,200 MHz (2,197 MHz read), profile `bootstrap` (262,144 x 4, chunk 4,096), int8 KV,
decode graphs on. **The number the user sees is the shipping configuration: `SPEC=dspark PREFIX_CACHE=on VISION=on`**, the prompts sent as a user sends them (no `prefix_cache: false`; every prompt of every size and pass is
a different filler stream, so none is served from the cache). `SPEC=none PREFIX_CACHE=off VISION=off` is the second table. Method as `2026-11-23-prefill-baseline` (`./inkling2mini.sh bench prefill-sweep`, four passes a size,
the first discarded, medians of three; the rate is `usage.prompt_tokens` over the server's TTFT; the decode rate follows the 128-token reply of the same request). **The 3,000 tok/s at 32K is not reached: 2,302.2.**

| Shipping configuration | Median tok/s | TTFT (s) | before (`1d53344`, no change of mine) | change | % of the bf16 ceiling | decode after it (tok/s, drafter on) |
|---|---:|---:|---:|---:|---:|---:|
| 8K | **2,435.6** | 3.364 | 2,399.9 (3.414 s) | +1.5% | 25.6% | 87.7 |
| 32K | **2,302.2** | 14.234 | 2,268.0 (14.448 s) | +1.5% | 27.1% | 86.3 |
| 128K | **1,760.2** | 74.465 | 1,739.0 (75.373 s) | +1.2% | 29.9% | 78.2 |
| 224K | **1,418.8** | 161.665 | not measured | | 31.4% | 59.6 |

| `SPEC=none PREFIX_CACHE=off VISION=off` | Median tok/s | TTFT (s) | the previous record (`1a36951`) | change | decode after it |
|---|---:|---:|---:|---:|---:|
| 8K | 2,542.6 | 3.222 | 2,493.3 | +2.0% | 50.6 |
| 32K | 2,344.7 | 13.975 | 2,304.4 | +1.7% | 46.0 |
| 128K | 1,777.4 | 73.743 | 1,754.9 | +1.3% | 34.2 |
| 224K | 1,429.5 | 160.457 | 1,415.2 | +1.0% | 27.3 |

The "before" of the shipping table is the merged head `1d53344` booted and swept with the same tool before any change (`raw/before-1d53344-sweep/`, three sizes, no 224K: 2,399.9 / 2,268.0 / 1,739.0 tok/s). The
shipping configuration costs 4.2% (8K), 1.8% (32K) and 1.0% (128K) against `SPEC=none` on the same engine: the drafter's per-chunk context and tap writes, the cache's 64-row tail pass at the cut (a pass of 64 rows
still streams every expert: about 100 ms, `[INFERENCE]` from the weights' 27 GB at 233.8 GB/s) and the vision tower's resident memory; the split between them was not measured.

The ceilings (`docs/ROOFLINE.md`): 9,525 / 8,481 / 5,894 tok/s at 8K / 32K / 128K and 4,519 at 224K (derived there); 3,000 at 32K is 35.4% of the bf16 ceiling. 3,000 tok/s at 32K is 10.9 s: eight chunks of 1,365 ms against
1,779 ms now, **a further 414 ms a chunk (23%)**.

## One 4,096-row chunk before and after (`nsys -t cuda`, `inkling2_loop_prefill_perf::prefill_curve`, ms; depth = tokens before the chunk)

`raw/profile-final-depths-0-8k-32k-128k.txt` (every kernel of the final tree), the before is `2026-11-23-prefill-baseline/raw/profile-by-kernel-depths-0-8k-32k-128k.txt` (`7da8c61`).

| Stage | depth 0 before / after | 8K | 32K | 128K |
|---|---|---|---|---|
| MoE gate/up GEMM | 448.5 / **397.6** | 439.2 / 392.1 | 429.0 / 389.7 | 448.5 / 391.6 |
| MoE down GEMM | 213.5 / **186.2** | 206.1 / 182.6 | 205.8 / 179.8 | 211.3 / 181.5 |
| dense EXL3 GEMM | 243.3 / 237.6 | 243.5 / 236.0 | 239.8 / 236.1 | 250.4 / 238.4 |
| MoE `rot_in` | 110.9 / 110.5 | 110.3 / 106.7 | 106.2 / 108.2 | 111.6 / 110.4 |
| MoE combine | 74.0 / 74.1 | 74.4 / 74.1 | 74.2 / 74.3 | 74.2 / 74.4 |
| loop controllers (pair + blend) | 118.0 / 119.1 | 119.4 / 118.8 | 117.2 / 118.5 | 118.6 / 119.0 |
| bf16 linears | 96.7 / 97.3 | 97.4 / 95.7 | 89.8 / 96.9 | 95.8 / 96.4 |
| AttnRes mixes / short convs | 64.6 / 26.3 -> 65.3 / 26.6 | | | |
| bias tables / prologue | 53.5 / 34.6 -> 52.7 / 33.8 | | | |
| attention core, 40 sliding visits | 90.5 / 88.3 | 98.4 / 98.4 | 102.5 / 98.4 | 99.8 / 99.2 |
| attention core, 8 global visits | 31.1 / 30.6 | 126.5 / 124.1 | 404.9 / 403.6 | 1,543.9 / 1,523.1 |
| **span of the chunk** | **1,671.7 / 1,586.2 (-5.1%)** | 1,760.6 / 1,671.6 | 2,013.2 / 1,949.8 | 3,210.1 / 3,096.3 |

## The 8,192-row chunk (shipping configuration, `PREFILL_CHUNK=8192`; `raw/c8192/`)

| | 8K | 32K | 128K | longest stall of a decoding stream while a 32K prompt prefills |
|---|---:|---:|---:|---:|
| chunk 4,096 (default) | 2,435.6 | 2,302.2 | 1,760.2 | 1.97 s (eight gaps of 1.65-1.97 s, one a chunk) |
| chunk 8,192 | 2,479.3 (+1.8%) | 2,334.9 (+1.4%) | 1,789.3 (+1.7%) | 3.80 s (four gaps of 3.26-3.80 s) |

(`bench/tools/stall_probe.py`: a streamed 600-token completion with the drafter, a 32K cold prompt sent after 20 events; the stream's steady gap is 42-43 ms either way.) The 8,192-row chunk is exact (rows
are independent: `exactness.py` replies sha256 `f7975a6c...` at both widths, the 135,168-token prompt's last row digest equal) and 1.4-1.8% faster, and it doubles the time a decoding stream stands still behind a chunk:
it is **not** made the default (`PREFILL_CHUNK=4096` stays). Why it wins little: the MoE gains the expected 12% a row (the experts' weights, 0.57 GB a visit, are read once per chunk), but the dense large-row kernel read
every row group's A fragments once a strip and at 8,192 rows they no longer fit L2 (it took 1.9x as long a row until the strips were ordered fastest), and everything else is linear in the rows.

Memory: the 8,192-row boot plans `reserve_gib=6.42` and the same 4 lanes at 262,144 (`raw/c8192/`); lowest MemAvailable over the shipping boot and sweep 69,221 MiB.

## Commands (node-3 stands for the node's address)

```sh
./inkling2mini.sh build ce650e376cdeb8eff6b4d63f6ae63facc95a0d85 && ./inkling2mini.sh start               # shipping defaults
bench/records/2026-11-23-prefill-push-exact/tools/live.sh ship                                              # sweep (PREFILL_CACHE_DEFAULT=1) + the exactness checks + the stall probe
SPEC=none PREFIX_CACHE=off VISION=off ./inkling2mini.sh start && tools/live.sh nospec
PREFILL_CHUNK=8192 ./inkling2mini.sh start && tools/live.sh c8192
NODE=node-3 LAB_SRC=<engine worktree> PKG=gb-model scripts/ops_dev.sh build inkling2_loop_prefill_perf; bench/records/2026-11-23-prefill-baseline/tools/nsys_pf.sh deep2 PF_TOKENS=135168
```
