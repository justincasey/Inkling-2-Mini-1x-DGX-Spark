# Roofline on node-2 (plan 2.2, gate `roofline`)

`tools/roofline.cu` (with the four Inkling-2-Mini shapes added, labelled rows x K x N) compiled with nvcc on node-2 and run three
times on the idle node under `/tmp/rexl3-gpu.lock`, the SM clock locked at 2,200 MHz.

- raw output: `roofline.out`; clock, power and temperature samples every 0.5 s: `clocks.csv` (SM 2184-2190 MHz under load).
- `meta.json` carries the three-run means in `metrics` and the comparison with the `HW` table of `scripts/roofline.py` in `notes`.
- Result: read bandwidth, bf16, fp16 and fp8 are all within 3 % of the table (no edit proposed). The Mini shapes run at
  80-82 TFLOP/s bf16 (4096-row dense projections) and 49-57 TFLOP/s (256-row expert GEMMs); the latter are launch- and
  tile-bound, not a bf16 ceiling.
