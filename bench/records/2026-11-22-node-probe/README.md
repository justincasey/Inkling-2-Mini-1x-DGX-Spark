# Node probe, 2026-11-22 (plan step 2.1)

Read-only probe of the four Sparks, taken 2026-11-23T03:38-03:40Z, while they all hold the Inkling-2-Small TP4 group.
`meta.json` has the per-node facts (nodes named `node-1` to `node-4` in the order of the Small recipe's `.env.example` `NODES=`,
no addresses). The raw captures, the probe scripts and `commands.txt` (every command run on a node, verbatim) are in
`.work/small-state-2026-11-22/` of the main recipe worktree: git-ignored, because they hold lab addresses.

## Result in one line

The Small group is **up and idle**: 0 % GPU utilisation on all four, no request for 40 minutes (the last chat completion was
8.9 hours earlier), and every node reads about 11 to 13 GiB `MemAvailable`, so no node can take a 14 GiB pack write or a
build beside it. Using any node means stopping the whole group.

## Per node

| | node-1 | node-2 | node-3 | node-4 |
|---|---|---|---|---|
| Small rank (rank-N of `--tp-rank`) | 0 | 1 | 2 | 3 |
| GPU, driver | GB10, 580.178.04 | same | same | same |
| SM clock, idle sample (max 3,003 MHz) | 2,184 | 2,190 | 2,197 | 2,197 |
| GPU utilisation, GPU process | 0 %, 104,291 MiB | 0 %, 103,208 MiB | 0 %, 103,208 MiB | 0 %, 103,208 MiB |
| `MemTotal` (kB) | 127,598,644 | 127,598,640 | 127,598,640 | 127,598,640 |
| `MemAvailable` at probe (GiB) | 11.12 | 12.65 | 12.48 | 12.49 |
| `MemAvailable` in `status` a minute later (MiB) | 11,391 | 12,983 | 12,823 | 12,819 |
| guard minimum since boot (MiB) | 10,764 | 12,659 | 12,301 | 12,600 |
| swap in use (MiB) | 311 | 0 | 507 | 151 |
| disk free on `/` (holds `/home/spark` and `models/`; 3.7 TB) | 863 GB | 1.5 TB | 521 GB | 491 GB |
| containers | `other-service-1` | `other-service-1` | `other-service-1` | `other-service-1` |
| `sudo -n true` | works | works | works | works |
| nvcc | 13.0.88 | 13.0.88 | 13.0.88 | 13.0.88 |
| Rust | 1.89.0 | **none** (no rustup, no `~/.cargo`) | 1.89.0 | 1.89.0 |
| `hf` | `~/venvs/hf/bin/hf` | same | same | same |
| holder of `/tmp/rexl3-gpu.lock` | the rank (fd 9) | same | same | same |
| host uptime, load average | 22.0 d, 1.10 | 22.0 d, 1.01 | 22.0 d, 1.10 | 22.0 d, 1.06 |

## The Small group, captured

- All four ranks run `bin/rexl3-serve` sha256 `dd8bd302...e0a8`, commit `a73ff7a3fe1c...`, version 0.1.1, `cuda_arch sm_121`,
  up 16 h 24 m (started about 2026-11-22T11:16Z).
- Every rank's live command line and `REXL3_*` / `NCCL_*` environment equal the Small recipe's own `print-serve`, rank by
  rank (`--tp-ranks 4 --max-model-len 1048576 --slots 4 --kv-tokens 4194304 --prefill-chunk 4096 --kv-dtype int8 --spec dspark
  --prefix-cache on --allow-unvalidated-pack --port 8000`; vision tower on rank 0). The engine adds `REXL3_MALLOC_REEXEC=0` by
  itself after re-exec; that is the only extra key.
- Guards: one per node, armed on the rank's pid with floors 2,048 / 1,024 MiB, swap cap 64 MiB while `MemAvailable` < 8,192 MiB,
  page-cache drop every 20 s while loading; no `KILL` line in any guard log. The Small recipe is at git head `12337a2a...`
  with a clean tree.
- Activity (rank 0's log, the whole file: 63,602 lines): 63,530 are `http` events, almost all `/metrics` at 1 per second from
  some monitor; the rest: 120 `/v1/chat/completions` (the last 31,950 s before the probe), 23 `/health`, 14 `/v1/models` (the
  last 2,410 s before), 5 `404`. Ranks 1 to 3 print the same `tp_follower_stats` line over and over (applied=147887, prefills=319).

## My doctor, once, on node-1

`NODES=<node-1> ./inkling2mini.sh doctor` failed as it must: `GPU in use` (the rank, pid and command line named),
`/tmp/rexl3-gpu.lock is held` (the same pid), `MemAvailable 11 GiB < 100`, `pack: 13 files missing or short` (not staged yet);
the rest passed (GB10, flock, `sudo -n`; `engine none`).

## Nothing on any node was changed

Every node command is in `commands.txt`: `ssh -o BatchMode=yes -o ConnectTimeout=10 -n` with `uname`, `nvidia-smi` queries and
`-q`, `cat` of `/proc` files, `df`, `free`, `docker ps`, `sudo -n true`, `nvcc --version`, `command -v`, `rustup toolchain list`,
`ls -la`, `cat` of `run/*.pid` and `bin/rexl3-serve.commit`, `sha256sum` and `--version` of the Small's binary, `fuser`, `lsof`,
`pgrep`, `/proc/<pid>/{exe,cmdline,status,environ}` (environment filtered to `REXL3_*` / `NCCL_*` before it is printed), `ps -o
etimes` and one `ps -eo ... comm` (process names only), and filtered tails of the rank and guard logs. Beyond the listed commands:
`ps`, `grep` over a tail of the logs, and `ls -l /proc/<pid>/exe`. No kill, start, stop, write, download, build, `drop_caches`,
clock or power command. After the doctor a last read showed the lock file's mtime unchanged (2026-11-07), both pids unchanged,
and no Mini directory or pack on node-1. The doctor's one `flock -n -x` on the lock failed because the rank holds it, so
it took nothing.

## What restoring the Small needs

1. Our server stopped on the node first (`./inkling2mini.sh restore-other` does this), `/tmp/rexl3-gpu.lock` free.
2. `./inkling2.sh start` in the Small recipe (git head `12337a2a...`), which first runs its own doctor on all four nodes: GPUs
   idle and alike, lock free, `MemAvailable` >= its `MIN_AVAIL_GIB` (100 GiB) on every node, only `other-service-1` / `other-service-2` as
   containers, the pack at full size, fabric reachable, the pinned NCCL library, one engine commit and sha on all ranks.
3. The installed engine untouched: `bin/rexl3-serve` sha256 `dd8bd302...` and commit `a73ff7a3...` on all four nodes. Never
   rebuild; do not touch `BASE_DIR` of the Small (`bin/`, `src/`, `target/`, `lab/`, `run/memguard.sh`).
4. Boot time of the Small is up to `READY_TIMEOUT_S` (1,800 s by default); restore is done when `/health` is 200, `print-serve` is
   byte-identical to `small-print-serve.txt`, each rank's command line and environment equal this capture, the guard is armed on
   every rank and `bench/tools/smoke.py` of the Small recipe passes.
   (Added 2026-11-24, after this record was written: restoring now locks the ranks at the GPU clock cap first, `./inkling2.sh gpu-clocks 2200 && ./inkling2.sh start`, because the Small's `start` does not apply it; `docs/CLEANUP.md` has the procedure and `./inkling2mini.sh restore-other` runs it.)
5. Page cache: a 14 GiB pack on a node leaves cache that counts against the Small's 100 GiB gate; the Small's guard drops it
   while loading, `sudo -n` works on all four.

## Surprises and unknowns

- node-2 has no Rust toolchain at all: `install` would put rustup there, or choose another node for builds.
- The serving process shows as `exe` (the engine re-execs itself), so `pgrep -x rexl3-serve` finds nothing; only the anchored
  command-line pattern (`^BASE_DIR/bin/[r]exl3-serve( |$)`) works, as in both controllers. A bare bracket-trick pattern also matches
  the probing shell itself when the script travels on the command line.
- The idle SM clock (2,184 to 2,197 MHz against a 3,003 MHz maximum) fits a 2,200 MHz lock, but no query here proves it
  (`Applications Clocks` reads 2,418 MHz default): UNKNOWN.
- Not read, by rule: the Small recipe's `.env` (so its hash and `MIN_AVAIL_GIB` override, if any, are unknown here).
- The first probe round (`run1/`) used a process pattern that missed the rank; round 2 fixed it. Both are in `commands.txt`.
