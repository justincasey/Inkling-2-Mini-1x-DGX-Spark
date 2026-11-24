# Agent Instructions — Inkling-2-Mini-1x-DGX-Spark

This is a serving-recipe repository. A change is correct only when the pinned stack serves and a record says so.

## Read first

- `README.md` — lane, pack, pins, risks.
- `docs/FIT.md` — per-rank weights and KV math.
- `intake/intake.json` — manifest/census/footprint evidence.
- `docs/DECISIONS.md` — decisions and traps.

## Rules

- Pins in `config/pins.sh`; not tunable from `.env`.
- No invented numbers: a quoted number needs `bench/records/<date>/meta.json`.
- Fail closed: no flag that skips a gate; the memory guard is mandatory.
- The GPU SM clock cap (2,200 MHz, `GPU_CLOCK_CAP_MHZ` in `config/pins.sh`) is hardware protection: in place whenever any work runs on any node,
  never released, raised or changed; no reset, no A/B across clocks. Every tool that runs GPU work applies it first (`scripts/gpu_cap.sh`).
- Remote scripts receive variables by name (no `$1`).
- Merged serve config, never a quant block mounted over `config.json`.
- Run the static tests before any hardware run.

## Before finishing

```bash
bash tests/test_recipe.sh                            # static tests; the last line is `test_recipe: N passed, 0 failed`
python3 -m py_compile scripts/*.py
python3 scripts/roofline.py | grep -F '**14.000**'   # resident total equals the card's 14.00 GiB
```

List only commands that exist in the tree. The bench-tool tests (`test_gates.py`, `test_bench.py`) join this block when plan
step 5.5 writes the tools.
