# first-boot: Inkling-2-Mini on one Spark (gate `boot`)

Result **pass** (`meta.json`). Engine: lab build of the committed head `c0cf5a1ba9b0c05b85b16fe3b1d085081c06d470`
(branch `inkling2-loop-backend`, clean, `dirty` false), recipe `bootstrap` profile, node-4, GPU clock locked at 2,200 MHz.
`REXL3_ALLOW_UNGATED_COMPONENTS=1`: `inkling2_loop_short_conv` and `inkling2_loop_forward` are not in `BUILT_COMPONENTS` yet, the
engine names them at boot (`serve.log`, `event=ungated_components`); the outputs are unqualified until the oracle comparison.

Commands (`meta.json` has the exact serve line): `./inkling2mini.sh build c0cf5a1b...` (lab build, about 2 minutes warm on node-4),
`doctor`, `print-serve`, `start`, `memwatch report`, `stop`. The git-ignored `.env`: `NODES=<node-4>`, `SPEC=none`,
`PREFIX_CACHE=off`, `VISION=off`, `EXTRA_ENV=REXL3_ALLOW_UNGATED_COMPONENTS=1`.

| Fact | Measured | Raw |
|---|---|---|
| Boot to `/health` 200 | `boot_s=17` (engine `phase=ready t=11.96 s`; weights 13.55 GiB in 7.5 s, 24 layers) | `serve.log` |
| Capacity line | `slots=4 max_context=262144 kv_pages=4100 kv_tokens=1048576 kv_gib=16.14 lanes_at_max_context=4 weights_gib=13.55 free_after_gib=101.45 reserve_gib=5.00 state_mib_per_slot=166.3 kv_bytes_per_token=16512 spare_gib=71.66` | `serve.log` |
| Against `docs/FIT.md` | 16,512 B a token in int8 equals FIT.md's figure; the bootstrap pool without the drafter's plane is 4 x 262,144 x 16,512 B = 16.12 GiB (FIT.md's 18.14 GiB adds the drafter's plane); the planner rounds the pool up by one 256-token page a slot (4,100 pages, 16.14 GiB); weights planned 14,552,922,948 B, loaded 14,552,899,328 B (24 KB less); per-slot state 166.3 MiB (FIT.md's 0.63 GiB for 4 slots is 161 MiB a slot before the window records) | `serve.log` |
| MemAvailable | 115 GiB before (doctor), 78,952 MiB (77.1 GiB) at ready | `status-ready.txt`, `memwatch.txt` |
| Guard minimum | 78,795 MiB, at the first chat request after ready (the whole window: loading 78,985, ready 78,795); swap unchanged at 151 MiB | `memguard-min.txt` |
| GPU | the server process holds 39,617 MiB (nvidia-smi) | `status-ready.txt` |

Not decomposed: the 39.6 GiB of the process beyond weights (13.55) + KV (16.14) + slot state (0.65) is the CUDA context, the step
buffers (`memory::UNPRICED` lists five not yet priced) and the allocator; the plan holds back its 5 GiB `ACTIVATION_RESERVE`
separately. The next profile bracket (`long`, `max`) must read this figure, not FIT.md's ~5 GiB.

Host tests: `gb-model --lib families::inkling2` 76 passed (31 of 31 of the Small's among them), `gb-server inkling2_loop_bare` 1 passed.
