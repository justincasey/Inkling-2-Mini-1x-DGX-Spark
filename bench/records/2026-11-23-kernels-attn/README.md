# kernels-attn: attention at 24 / 8 / 3 heads (plan step 4.3, audit lane B)

Result: **pass** on node-3 from the committed engine tree `dbfa89a04f7adfeda8efed206945eccbe46bf450` (branch `inkling2-loop-attn`,
clean). Commits on that branch after the base `df4862b`: `9b52d3a` (int8 KV page header), `c2c46bf` (kernels, Rust API, CPU
reference), `dbfa89a` (tests, oracle fixtures, Mini golden hashes, release-notes fragment). Raw outputs: `raw/`. Counts per binary
and filter: `meta.json`.

## What the engine now has

| Piece | Where | Fact |
|---|---|---|
| Page header | `csrc/inkling2_kv8_mini.cuh` | `kv8m::Page<NKV>`: K or V page 262,144 B of values + 2,048 B of exponents = 264,192 B at 8 heads (528,384 B a layer page, 16,512 B a token over 8 global visits); at 2 heads it is the Nemotron page (66,048 B, `static_assert`). `nemotron_kv8.cuh` is untouched. |
| Kernels | `csrc/inkling2_attn.cu` | every kernel a template on `<KV, GR>` (2 x 6 the Small, 8 x 3 the Mini), chosen by `Args::nkv` / `Args::grp` |
| Prologue geometry | same, `Pro<KV, GR>` | a CTA serves 12 query heads and its KV heads: the Small's 16 warps = 512 threads unchanged; the Mini two CTAs a row (`blockIdx.y`) of 12 + 8 warps = 640 threads (the Small's map at 24 + 16 warps would be 1,280 threads > 1,024). Bias entries a thread: `ceil(1024 / threads)` = 2 at both. |
| Attention core | same, `Geo<KV, GR, GLB, RB>` | `NW = RB * GR / 16` whole m16 tiles: RB 8 / 16 (GQA 6), 16 / 32 (GQA 3); exponent copies `TILE * KV / 16` per K and per V (32 at 8 heads), V offset `KV * D`; shared memory 79,872 B (global, RB 16) to 92,160 B (global, RB 32), under the 99 KiB limit by `static_assert` |
| Commits | same | `ring_write` 256 threads x 16 B (a 2,048-element row), `conv_commit` 1,024 threads striding the 2,048 channels (the Small's 512 threads, one channel each, unchanged) |
| Log scaling | same + `Shape` | floor and alpha are call arguments (`Shape::{SMALL: 128000, MINI: 8192}`, alpha 0.1): `tau = fp32(1 + alpha ln(max((p + 1) / n_floor, 1)))` evaluated in f64 as before, which is NUMERICS.md's `log_tau` |
| Rust API | `src/inkling2_attn.rs` | `Shape` carried by `ConvState`, `Ring`, `Scratch`, `PagedKv` (`Kv::Paged`); every Small constructor keeps its signature and means `Shape::SMALL`; `Rows::upload_for(shape, ..)`; `VisitState` holds conv state, window record, ring (sliding visits) and page pool (global visits) per VISIT; sizes: `Shape::{slot_bytes 12,288, bias_row_bytes 49,152, partial_bytes 12,480, page_bytes 264,192}`, `Scratch::bytes_for_shape` |
| CPU reference | `src/inkling2_attn_ref.rs` | shape-generic (`RefWeights::shape`, `RefSlot::for_shape`, `tau_for`); same operation order |

Why the arithmetic does not depend on the geometry (also the comment in the `.cu` header): a launch shape only decides which thread
computes which element; an element's arithmetic (butterfly norms over a warp's 4 x 32 values, r's sequential 16-term sum, 16 fused
multiply-adds a bias entry, the conv tap order, the m16n8k16 k-order of the score and P.V products, the online softmax per
64-key tile, the chunk folds in ascending order) is a function of its own operands. So the Mini keeps the Small's tile (64) and
chunk (512 global, 256 local) sizes, and **the oracle's `Emulation` knobs (`tile 64`, `chunk_global 512`, `chunk_local 256`) are
already the Mini kernels'**; no knob changed, `oracle/tests/test_emulation.py` pins the defaults.

## What is proven (all on node-3, from the clean HEAD)

- **Small regression**: `inkling2_attn small::` 14 passed (12 real tests, 2 return at once at the Small) and `inkling2_attn_oracle
  small::` 2 passed; the golden hashes of `testdata/gpu/golden_inkling2_attn.txt` are unchanged (the file is not edited), so the
  Small's outputs and state are bit for bit those of before the template change. `gb-model` `families::inkling2::` 31 of 31 (Mac).
- **Header / page layout**: `mini::prologue_matches_the_cpu_transcription_bit_for_bit` reads the K and V pages at 264,192-byte
  strides (token stride 1,024 B, exponents after 262,144 B) and equals the CPU transcription for every row of prefill chunks, decode
  rows and windows, with a shuffled block table; `mini::kv8_page_is_264192_bytes_at_the_mini`.
- **Geometry**: every kernel launches at 24 / 8 / 3 (prologue 640 threads x 2 CTAs, core 96 / 192 threads, ring write 256, conv
  commit 1,024); prologue (q, kv, bias, int8 pages, conv records) is bit-equal to the CPU transcription; attention within
  6.1e-3 (local, global) of the f64 reference (`mini::*attention_matches_f64*`, 4.5e-3 across the log-scaling floor at 8,187).
- **Row block 16 against 32**, partial against fold mode, chunking, serial decode, neighbours: `mini::row_block_size_changes_no_bit`,
  `mini::chunked_prefill_equals_whole_equals_serial_decode_with_neighbours` (row blocks alternate), `mini::partial_mode_equals_fold_mode_past_256_chunks`.
- **Per-visit state**: `mini::per_visit_state_follows_the_loop_schedule`: 48 visits (8 global, 40 sliding), 12,288-byte conv slots,
  rings of 168,919,040 B for 40 visits, per-visit rows and one `conv_commit_layers` launch over 48 layers, every visit's output and
  final conv state against its own CPU slot.
- **Boundary and rollback**: `mini::boundary_1020_to_1040_ring_and_page` (ring row `p mod 1031` equals the prologue's k/v row, int8 page
  bytes and exponents equal the CPU's, outputs within tolerance across 1,023 / 1,024), `mini::verify_windows_with_rejected_rows_equal_serial_replay`
  (windows of 8 rows, rejected rows, conv commit with `keep`), `mini::graph_replay_equals_eager`.
- **Oracle at 24 / 8 / 3**: fixtures `testdata/inkling2_loop/attn_oracle_{local,global}.bin` from `tools/inkling2_loop_attn_oracle.py`
  (global with the oracle's `kv8` emulation): kernels within 7.8e-3 max relative error and cosine 0.99999 of a head; the CPU transcription
  differs from the oracle in 22 / 6 of 347,136 outputs (fp32 against float64 rounding of a prologue value).
- **Real tensors** (`real/attn_real.bin`, 16 real tokens, layer 0 local and layer 11 global, oracle forward of visits 0..11 on the real
  pack, `real/README.txt`): layer 0: q_hat, k_hat, v_hat bit-equal (0 differ), the bias table 1 of 393,216 elements 1 ulp, attention vs the
  oracle's `attn_bf16_p` emulation 3 of 49,152 elements, 1 ulp; layer 11: kv_hat 0 differ, the int8 K and V values (dequantised) 0 of 16,384
  differ at P0 = 0, 8,180 and 300,000 (log scaling tau up to 1.36), q_hat 2 of 49,152 differ by 1 ulp (the CPU transcription has the
  same 2), bias 0 / 4 of 393,216, attention vs the emulation 29 of 49,152 elements (all within 1.6e-3 of the head's largest output).

## Not proven / UNKNOWN

- **Speed is not measured** (no timing record; `ops_dev.sh` cannot pass `--ignored` to the ignored `bench` test). Register use of the
  640-thread prologue (102 registers a thread allowed) was not inspected; the tests show it is correct, not fast.
- Attention outputs are not bit-equal to the oracle's emulation (the kernel sums in fp32 with `ex2.approx`, the oracle in float64 with
  bf16 P): 3 and 29 of 49,152 elements differ, a few ulps of a near-zero output at worst (76 ulp of itself, 1.6e-3 of the head's maximum).
  Per the NUMERICS.md principle the oracle is the tolerance anchor.
- Real attention at P0 > 0 (log scaling in the attention output) was not compared: the oracle's fresh cache holds only the 16 rows
  while the kernel also attends to every earlier (zero) position; log scaling is checked at the q / r level (q_hat, bias) with real
  tensors at tau = 1.36, and in the attention of the synthetic tests past the floor.
- The log-scaling formula is NUMERICS.md's (f64 evaluation, one fp32 rounding); HFL evaluates in fp32. A last-bit fp32 difference of tau
  would flip a bf16 rounding of `q_hat * tau` only rarely; not measured. The 2 differing `q_hat` elements on the real layer 11 are at
  tau = 1 and 1.36 alike, so they come from the clip's float64 against fp32 arithmetic, not from tau.
- Not run: 1M-context decode, a 4,096-row Mini prefill timing, other row-block sizes (RB 8 at GQA 3 is not instantiated), the drafter's
  attention (`inkling2_dspark.cu`, step 4.6), `inkling2_ops`.
- Host wiring is not mine: the lead / EngineHost must build `Scratch::with_shape(MINI, ..)`, `Rows::upload_for(MINI, ..)` (row block 16 or 32;
  the Small's default 8 panics at the Mini), `VisitState` per slot group, and price `Scratch::bytes_for_shape(MINI, rows, cap)` in
  `memory::UNPRICED`: 49,152 B of bias table + 6,144 B of q + 4,096 B of k/v a row, 12,480 B a chunk partial, plan `(4 + 6 rows) * 4` B.

## Node 3 footprint (this lane)

`/home/spark/inkling-2-mini-1x-dgx-spark/lab/rexl3-inkling2-loop-attn/` (1.1 GiB: `src` copy, `target` build, `build.*`, `bin.path`, `real/attn_real.bin`
6.2 MiB, `golden_loop_attn.txt`); remove with `NODE=<node-3> LANE=rexl3-inkling2-loop-attn scripts/ops_dev.sh clean`. No process left running.

## Files here

`raw/*.txt` the four final runs and the node status; `real/attn_real.bin` + `real/README.txt` the real-tensor fixture; `tools/real_dump.py`
its generator (needs the pack by HTTP range reads or a local copy; `--stage capture` then `--stage emit`, `--check FILE`).


## Public copy

`real/attn_real.bin` (6,392,859 bytes, sha256
`23ee74d4cfb826acfc57b940839b4346e84a495bb7da943cc3f8987b30b677e5`) holds real weight tensors of the pack and values computed on them. It is not in this
public repository (no model weights or derivatives are published). `real/README.txt` documents its layout; `tools/real_dump.py` regenerates it from the public pack.
