# prefix-cache: cached == uncached on one Spark (gate `cache`, plan step 4.5)

Result **pass** (`meta.json`). Engine lab build `de2d4a30e423a6e671b797cc6c0b9c1848703c27` (branch `inkling2-loop-state`, clean, `dirty`
false; the branch tip after it adds only a rustfmt re-wrap of the snapshot module and the two test files), recipe `5c62b03`, node-3, GPU
clock locked (2,197 MHz read), profile `bootstrap` (4 x 262,144), `SPEC=none`, `VISION=off`, `PREFIX_CACHE=on`. The first two GPU runs were
made from the working tree that the commits `5f2d5f2` (the feature) and `de2d4a3` (the tests) hold, before the last was committed.

What this record proves is what a snapshot of the Mini at `len` tokens must hold, that a restore from it is byte-exact, and that the
engine's cache (index, cut planner, budget, eviction) around it gives byte-identical replies through the real server. What it does not
prove is in "Not proven".

## What a snapshot holds (derived from `model.rs` forward and `VisitState`, then proven below)

The forward's AttnRes blocks, the message bank and the prelude's partial block are locals of one step (`Inkling2LoopRank::forward`: `blocks`,
`prelude`, `bank` are built from the step's own rows), so they are not state. Everything one token carries to the next is:

| Part | Where | Bytes at the Mini | In a snapshot |
|---|---|---:|---|
| K and V of the 8 global visits, int8 plus one exponent byte per (token, KV head) | the page pool, one page = 256 tokens x 8 visits x K+V = 4,227,072 B | 16,512 B a token | **by reference**: one reference to each of the `ceil(len / 256)` pages of the slot's table (all 8 visits address the same page ids); a restore shares the full pages and **copies the partial last page** (the 8 visits' K and V, 4.03 MiB) |
| the 40 sliding visits' rings, bf16 K and V of 8 KV heads, position `p` at row `p mod 1031` | per slot, `[1031][4096 B]` a visit | 168,919,040 B | **copied**, whole (the last 1,024 positions are the ones a later row reads; 7 slack rows and any row not written yet go along and are never read: `begin_slot` does not clear a ring either) |
| the 3 raw inputs of the K/V stream's short convolution, 48 visits | per slot `[3][2048]` bf16 a visit | 589,824 B | **copied** |
| the 3 raw inputs of the attention-output and MoE-output short convolutions, 48 visits | per slot `[3][2304]` bf16 each | 1,327,104 B | **copied** |
| the slot's length and block table | host | 16 KiB | the length is the snapshot's `len`; the table is the pages above |

One snapshot slot is therefore **170,835,968 B = 162.92 MiB** (rings 168,919,040 + conv 1,916,928), fixed whatever `len`: the engine's budget
counts whole slots, and the plan priced exactly this (`memory::snapshot_bytes`, checked against the rank's own layout at
`enable_snapshots`). The live information at `len` >= 1,024 is 40 x 1,024 x 4,096 + 1,916,928 = 169,689,088 B (the slack rows are 0.67 %
of the slot); at `len` < 1,024 the live rows are `len x 163,840` B of rings (len 256: 40 MiB) and the rest of the slot is copied
along unread. A compact layout for short cuts would save up to 87 % of the slot at 128 tokens (20 MiB of live rows) but needs variable-size slots in the engine's
budget; not built. Pages cost `ceil(len / 256) x 4,227,072` B held by reference (shared with the donor, no copy): 1,024 tokens 16.1 MiB,
12,000 tokens 189 MiB, 73,167 tokens 1.13 GiB.

Layout decision: a separate arena of snapshot slots (one device buffer, `slots x 170,835,968` B, allocated once at boot) and pages in the
KV pool beside the admitted ones (`kv_pages = kv_blocks + cache.page_blocks`), both carved from `spare_gib` by the window rule
(`plan_memory_cached_windowed`): the cache never takes the KV pool's share. At the bootstrap profile (window 262,144, 4 lanes) the rule asks
for `C` = 1,024 pages (one full-window entry), `N = max(8, 4 x 4)` = 16 snapshot slots and `C + 4 x ceil(C / 8)` = 1,536 pages:
16 x 162.92 MiB + 1,536 x 4.03 MiB = 9,226,158,080 B = **8.59 GiB**, reach 393,216 tokens, out of `spare_gib` 71.66.
Cut policy: the engine's `CutPolicy::interactive()` (`min_cut` 128, `tail` 64, `ghost_min_gain` 256, `min_gap` 128; no `cut_align`: a restore
inside a page copies the partial page and is exact there, see below), the Small's service pattern (chats growing by tens of tokens, a long
document asked about again). Arithmetic: a cold prompt pays about 0.08 to 0.10 s more (the slice ends at the cut, so the last 64 rows run
as a pass of their own; the copy itself is 2.1 ms, `-speed` record), and a hit saves 0.81 ms a token (cold prefill 1,237 tokens/s at 12K),
so a cut breaks even at about 125 tokens saved: `min_cut` 128 is at the break-even of one hit, the end cut of a prompt that is asked again
pays back many times over (4K: 22 x, 12K: 63 x, 32K: 154 x in TTFT).

## Proofs

**Rank, on the real pack, bit for bit** (`crates/model/tests/inkling2_loop_prefix.rs`, `raw/gpu-*.log`; the lines starting `LOOP_PFX` are the
measured values):

1. `prefix_exact_at_cuts`: 20 cuts, each a cold prefill of `cut + 200` tokens on slot 0 that snapshots at `cut`, then the restore into slot 1 and
   the rest of the prefill on **another chunk grid** than the cold one. Compared: the restored state at the cut against the cold slot's
   (every sliding visit's last 1,024 ring rows, the int8 K and V of every position of the 8 global visits, the conv state of all 48 visits:
   `read_slot_state`), the last row's top-64 candidates after the rest of the prefill, the state after it, the **full logits plane** of the first
   decode step (200,058 floats, bit for bit) and three more decode steps, and the pages and snapshot slots back to the baseline. Every
   comparison equal at all 20 cuts; the digests are the first 8 bytes of sha256 of the cold run's values:

| cut | cold chunk | resumed chunk | last row | state | decode |
|---:|---:|---:|---|---|---|
| 1 | 2048 | 129 | cdce5f343b13d39c | b0a651cd9e50a585 | d194be2f61a6b1a0 |
| 255 | 700 | 129 | d879929334a0929b | b85d818cf7e66ae4 | 3672326076fa0e6f |
| 256 | 700 | 7 | bffa2759f018dda3 | 8550e8bbc3fd7a57 | e0bdf9d081bbb5f9 |
| 300 | 2048 | 300 | 0f7f6de8ab703e42 | 72900afd094f8419 | c7074f1d78d67e11 |
| 512 | 129 | 1 | 58f723cc5c3db2aa | c028b1eac09cd8f3 | e0074b6225adca9d |
| 1023 | 700 | 129 | d7eb5c89000822b3 | ea72bf0edb2fa910 | e99ef3c21f5584f2 |
| 1024 | 2048 | 300 | cabcf89dc2d01a96 | d756860f9f0c6db4 | 5de27757d23a14a0 |
| 1025 | 300 | 7 | 2e076611d5bb72e5 | bba45c0479c9efda | 0ca6c3d2329bfe16 |
| 1030 | 700 | 129 | 7beb528c206e1e96 | 5663bc3b40eceaa4 | ac849cb743d6d2dc |
| 1031 | 700 | 129 | 042701665506f8ab | 13a0d41c1f183176 | 7fac5443a572a2c9 |
| 1032 | 2048 | 1 | 249e52b48ff9c655 | 285a5e0501d9db44 | edaf5edaee5d4895 |
| 1100 | 300 | 129 | b439a55fd0863fec | 76501d4c3f8b2903 | 22afd449db5d24e7 |
| 1500 | 129 | 700 | 920cf083880902b7 | beefd772056e8ed8 | bf903d300f33153c |
| 2047 | 2048 | 129 | 5d8e85c1a136c52e | af20e60d2425ea4f | e880b31871e6df24 |
| 2048 | 700 | 300 | e4cc70379c583b72 | ade780c6340613cc | 6a6029cbd3ff9d1b |
| 2049 | 300 | 129 | 2cf093f80fbf9403 | d7f72d51f3351596 | edf1cb54720e3627 |
| 2062 | 2048 | 7 | 4ec1f7d6e14d28a9 | 5a16ac5bf8beb0a0 | 26d151b3c5bf3908 |
| 2063 | 700 | 129 | bd829e568ff2ff1a | 5230f2056674da93 | 9dedb85bbe155e91 |
| 3000 | 2048 | 300 | cd9bea0615348eac | 2fdbf18cfccf4c30 | aa8577c08a9aaf9e |
| 4096 | 700 | 129 | 6d5f3ff41ad61778 | d56153549fb108d3 | f19731f673b4e115 |

   Below the window (1, 255, 256, 300, 512, 1,023), at it (1,024), above it (1,025 to 4,096), around the ring's wraps (position 1,031 is row 0
   again: cuts 1,030, 1,031, 1,032; 2,062 is the second wrap: 2,047 to 2,063), page boundaries (256, 512, 1,024, 2,048, 4,096) and inside
   pages. A cut is never on both grids, so every snapshot is taken mid-prefill of the cold run and every resume starts off the cold grid.
2. `prefix_logits_planes_equal`: the next prefill chunk (40 rows) after a restore at cuts 300, 1,024, 1,500 and 2,048, from the oracle dump
   (`REXL3_INKLING2_LOOP_DUMP`): the planes `hidden` (the partial block entering each of the 48 visits, 9,031,680 B), `core_out` (368,640 B),
   `h_final` (184,320 B) and `logits` (32,009,280 B) are **byte-equal** between the cold slot and the restored one (digests in the log).
3. `prefix_restore_after_eviction`: the donor's slot is reused by an unrelated 1,600-token prompt (its ring overwritten, its pages recycled), a
   second snapshot is released (an eviction), a snapshot slot is reused by a new snapshot, and three restores from the survivor (cut 1,092,
   inside a page, above the window) each give the cold state, last row, state after the prefill and 4 decode steps. The comparison is not
   vacuous: a restore from the other prompt's snapshot at the same cut gives another reply (asserted). Every page and snapshot slot comes
   back (400 pages free again); a released snapshot cannot be restored ("not held").
4. `prefix_restore_beside_decoding_streams`: a restore and the 300-token rest of its prefill (in 100-row chunks) and a cold prefill of a third
   prompt between the single-row windows of a decoding stream, then windows of all three streams together: each stream's 12 decode steps equal
   its solo run bit for bit (the first stream's first 12 of its 17).
5. `prefix_long_prompt`: a 12,300-token prompt cut at 12,000: restored state (200 MB of KV compared), last row, state, decode equal.

**Engine, through the real index and planner, on the host** (`raw/host-*.txt`): `gb-model --lib families::inkling2_loop::` 70 passed, the Small's
`families::inkling2::` 31 of 31 unchanged, `family::` 39, `plan::` 66, `serve::` 34. Six new tests drive the real `Engine` over a simulated rank
with the Mini's state structure (pages through a real `KvTables`, a 1,031-row ring never cleared and started from garbage, a conv chain):
cached replies equal uncached ones at 9 prompt lengths around the window and both wraps (greedy and sampled), on 6 chunk grids (1 to 4,096
rows), with concurrent streams, after eviction (releases exactly once, pages and slots balance), a failed snapshot and a failed restore
(the reply stays exact, the entry is dropped); every part of a restore has a mutant (no ring, no partial page, no conv state) that must
change the reply and does. Six more tests pin the backend's slicing and bookkeeping against a recording rank (a slice ends at the first cut, a
snapshot with no free slot or a recoverable error is skipped, a fatal one fails the slot, a restore starts the slot at the prefix and a failed one
ends it, a handle is released once). Plan tests: 16 snapshot slots, 1,536 pages, 8.59 GiB, reach 393,216 at the bootstrap profile;
`--prefix-cache on` with no room refuses by name; `auto` with images is no cache; `on` or a size with images, a drafter and
`--prefix-cache pool` are refused by name.

**Live, `gates.py cache`** (`raw/gate-cache/`, PASS): per prompt the uncached reference, then cold and warm with the cache: replies identical
for the five prompts (29, 534, 2,043, 8,347 and 73,167 tokens); 0 cached tokens on every uncached and cold reply; the warm reply served from
the cache for every prompt above the smallest cut (470, 1,979, 8,283 and **73,103** tokens; the 29-token one is below it); the two
shared-prefix pairs (2,081 and **73,246** shared tokens, different tails): both uncached, then both cached, the second served 2,026 and
73,191 tokens from the cache, all equal.

**Live, `bench/tools/cache_stress.py`** (`raw/cache-stress.json`, PASS): 24 distinct 1,500-token prompts against 16 snapshot slots: each
prompt's uncached reply, then each with the cache on (cold), then again in reverse order: 16 restores (prompts 8 to 23, still stored) and
8 cold misses (0 to 7, evicted), 30 evictions in all, every reply equal to its uncached one, 0 restore failures. Then one stored 3,000-token base
and four requests at once on four slots (three sharing the base, each served 2,960 tokens from the cache, and one unrelated cold), 128 tokens
each: every reply equals its solo uncached reply.

## Not proven

* Images with the cache: refused by name (`start` and the engine), not built.
* A drafter with the cache: the Mini serves `--spec none`.
* A restore after a CUDA graph replay of the decode windows: that lane (GraphLane) is a separate branch; its graphs do not touch the state
  the snapshot copies, but the combination is not run here.
* Boot to boot (the same bytes across two server runs) is the gate's job in 6.4 and is not claimed by this record.
* The server crate's own test `inkling2_loop_bare_model_flag...` fails on this branch and on its base: it still asserts that the tower is
  not built (`!a.vision`), which the vision lane made false. Not touched here.

## Files

`raw/gpu-*.log` (the node's test output), `raw/host-*.txt`, `raw/gate-cache/` (`cache.json`, `cache.log`, `gates.txt`), `raw/cache-stress.json`,
`raw/serve-start-cache-on.log` (the capacity line `prefix_cache=on prefix_entries=16 prefix_pages=1536 prefix_gib=8.59
prefix_reach_tokens=393216`), `raw/status-cache-on.txt`, `raw/metrics-prefix-cache-on.txt` (hits, evictions and the cache's bytes after the
runs). Timings and memory: `../2026-11-23-prefix-cache-speed/`.
