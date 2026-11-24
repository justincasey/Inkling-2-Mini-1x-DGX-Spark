# process-tests-linux: gb-server's process tests on Linux (gate `process-tests`)

Result **pass**: 14 of 14 (`meta.json`). `scripts/ops_dev.sh test process ""` on node-4 (idle, GPU capped at 2,200 MHz, the cap's success line twice in `raw/process-linux.log`),
engine `59d08a9c5e9f9d3ce96ed9d2353c68b70169cc1a` (the pin), package `gb-server`, test binary `process`.

These tests start the real `rexl3-serve` binary against a mock engine and check what an operator relies on: exit codes (0 on an idle SIGTERM, 70 on a fatal backend error, SIGABRT from the
watchdogs), a drain inside the grace period, no ready-looking health while loading, a bounded failure when the owner is wedged, and the malloc-arena re-exec (`MALLOC_ARENA_MAX=2`).
They need Linux (`/proc`): on macOS the arena test fails at `crates/server/tests/process.rs:425`, which is why the record is made on a node.
