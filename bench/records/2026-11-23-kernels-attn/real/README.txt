attn_real.bin: real-tensor attention fixture, Inkling-2-Mini, global layer 11 (visit 11, pass 0) and local layer 0 (visit 0)

File     real/attn_real.bin, 6,392,859 bytes, 74 entries
sha256   23ee74d4cfb826acfc57b940839b4346e84a495bb7da943cc3f8987b30b677e5
Source   tools/real_dump.py (this directory's parent), stages capture + emit; the file is bit-identical on a re-run of emit.

Prompt (T = 16 tokens, no special tokens added, the pack's tokenizer.json at the pinned revision):
  "The quick brown fox jumps over the lazy dog near the quiet river."
  ids = [371, 157021, 161319, 220, 32329, 160581, 82, 150053, 149876, 161703, 153350, 152836, 149876, 168268, 158897, 13]

How it was produced
  Pack: vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw at revision 44c78c9d229b43dd45e41506071b9f5f939bed44, read by HTTP range
  reads (public GET, no credentials) into a sparse scratch copy outside the repo (2.68 GB fetched, 532 requests).
  capture: the recipe oracle `Inkling2LoopRef` (oracle/inkling2_ref.py, default Rounding, default exact Emulation: kv8,
  attn_bf16_p, moe_fp16 all off), forward over the 16 tokens through visits 0..10 in full and visit 11 up to its attention
  core (then stopped). Dense EXL3 weights (attention, router) and all routed experts decoded exactly by the oracle's numpy
  decoder oracle/exl3.py (bit-identical to the engine-side CPU decoder per oracle/records/exl3-real-verify.json); the Rust
  CLI exl3dq was not used. For visits 0 and 11 it records the visit's input P, x_q, x_k, x_v, the six projection outputs of
  attention() (captured from the oracle's own `lin` calls) and the weights as stored, and the oracle's own `core` tap.
  emit: re-runs the body of `Inkling2LoopRef.attention` step by step on those inputs by calling the oracle's functions
  (static_conv / dynamic_conv, l2clip, log_tau, relative_bias, kv8_roundtrip, attention_core, attention_core_tiled, Rounding),
  for P0 in {0, 8180, 300000}: positions P0 + arange(16), fresh LayerCache (zero conv state, k_start = P0), same inputs.
  Checks done while emitting: the stepwise result without kv8 equals the oracle forward's `core` tap bit for bit (both visits,
  P0 = 0); the dyn_k / dyn_v taps are exactly what dynamic_conv consumes; every bf16 entry is exactly bf16-representable.
  Wall time on the Mac (10 cores, 6 torch threads, niced): capture 190 s total (forward 183 s, of which the 12 visits took
  17.0 14.2 16.0 17.1 15.5 15.0 12.6 12.2 13.2 13.3 11.4 4.4 s, mostly fetch time), emit 0.07 s.

Container (little-endian): magic "I2LR", u32 version = 1, u32 n_entries; per entry u16 name_len, name (ascii), u8 dtype
(0 = bf16 as u16 bits, 1 = f32, 2 = i32), u8 ndim, u32 dims[ndim], raw data row-major. `real_dump.py --check FILE` lists them.

Entries (prefix l0 = local layer 0, g11 = global layer 11; T = 16)
  meta.tokens i32 [16]; meta.p0s i32 [3] = 0, 8180, 300000
  <pre>.w.q_scale [24,128] (q_l2_clip.scale)   <pre>.w.r_scale [24,16] (rel_l2_clip.scale)   <pre>.w.rel_proj [16,1024]
  <pre>.w.k_conv, <pre>.w.v_conv: l0 = k/v_sconv.weight [1024,1,4]; g11 = k/v_sconv.static_w [4,1024]  (all as stored, bf16)
  <pre>.x.q [16,3072]  .x.r [16,384]  .x.gate [16,3072]  .x.sink [16,24]  .x.k [16,1024]  .x.v [16,1024]  (raw projections,
     bf16, = the oracle's q, rr, gate, sink, k_raw, v_raw)   .x.hidden [16,2304] (the visit's input P)
  g11.x.dyn_k, g11.x.dyn_v [16,32]: bf16(x_k . k_sconv.weight_proj^T), index head*4 + lag (x_v likewise)
  <pre>.p<P0>.q_hat [16,24,128] (after clip; g11 also times tau)   .r_hat [16,24,16] (likewise)   .k_hat, .v_hat [16,8,128]
     (bf16, before kv8)   g11 only: .k_rt, .v_rt [16,8,128] (after kv8_roundtrip)
  <pre>.p<P0>.bias [16,24,1024] for P0 = 0 and 300000 only: bf16(r_hat . rel_proj) BY DISTANCE (index = q_pos - k_pos)
  <pre>.p<P0>.out_exact [16,3072]: attention_core on the kv8 read-back K/V (g11) / bf16 K/V (l0), exact softmax
  <pre>.p<P0>.out_emul  [16,3072]: attention_core_tiled, chunk 512 (g11) / 256 (l0), tile 64, bf16 P, kv8 on for g11
     (both are bf16 of the core output before wo, like the `core` tap)

Sanity (P0 = 0): max |out_emul - out_exact| / max |out_exact| per head, max over heads (T = 16 tokens, 128 dims per head):
  l0 5.2e-3 (heads 0..23: 5.2e-3 3.8e-3 3.5e-3 2.3e-3 1.1e-3 2.5e-3 5.4e-4 1.8e-3 1.3e-3 8.4e-4 2.5e-3 9.9e-4 2.9e-3 1.7e-3
     3.7e-3 8.9e-4 2.6e-4 1.2e-3 8.0e-4 2.3e-4 2.9e-3 2.1e-3 4.7e-3 4.1e-3)
  g11 5.8e-3 (2.9e-3 2.8e-3 9.5e-5 4.0e-3 1.5e-3 2.2e-3 4.8e-3 3.2e-3 2.7e-3 3.4e-3 5.1e-3 5.8e-3 4.2e-3 5.0e-3 1.6e-3 2.7e-3
     1.8e-3 5.2e-3 3.0e-3 4.1e-3 4.7e-3 3.2e-3 2.2e-3 5.0e-3)

Note: at P0 = 8180 the global visit's positions are 8180..8195, so only 8192..8195 have tau > 1 (1 + 0.1 ln((pos+1)/8192) <=
1.0000122, fp32); after the oracle's bf16 roundings q_hat and r_hat of P0 = 8180 equal those of P0 = 0 for all 16 tokens. At
P0 = 300000, tau = 1.3601..1.3602 and q_hat / r_hat change accordingly.
