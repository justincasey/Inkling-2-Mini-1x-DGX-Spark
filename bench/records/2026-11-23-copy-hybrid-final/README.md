# copy-hybrid-final: the confirm run of the copy hybrid on the final engine head (gate `speed-decode`)

Engine `f4922eb462ef2d2b27e20024e0fc39bb7673cb11` (branch `inkling2-loop-hybrid`: the copy-hybrid commit `efcb6ed` merged with the dev head `be0ba1e` (DensePipe and the scrub); the serial-window chain tried in between is reverted: no code of it remains), node-4, TP1, shipping defaults, GPU cap 2,200 MHz applied by `scripts/gpu_cap.sh` before the run.
One fresh server, `bench decode` x3: **`decode_mean4_tok_s` 85.54** (85.54, 85.56, 85.36); the record `copy-hybrid` measured 85.41 (off 83.92, +1.49) on `efcb6ed`. `raw/on` holds the whole driver run (specmix `PASS`, `bench rolling` x3, specbench c = 4). The 100 tok/s target is not reached.
Exactness of this very commit: `copy-hybrid-final-exact`. The off column is not rerun here (the same switch gives the engine without the hybrid; see `copy-hybrid`).
