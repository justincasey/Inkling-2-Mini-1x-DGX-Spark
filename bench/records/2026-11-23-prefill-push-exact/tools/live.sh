#!/usr/bin/env bash
# live.sh MODE: the live checks of the prefill push on node-3, from the recipe worktree.
#   ship   shipping configuration (the .env defaults: SPEC=dspark PREFIX_CACHE=on VISION=on, chunk 4,096): exactness record + diff against the 7da8c61 record of
#          `2026-11-23-prefill-exact`, bench.py --ref PASS pin, specedge.py, exactness concurrent, the cold-prefill sweep (8K 32K 128K 224K, prompts sent as a user does),
#          the stall of a decoding stream while a 32K prompt prefills
#   nospec SPEC=none PREFIX_CACHE=off VISION=off: the sweep alone (comparable to the earlier records)
#   c8192  shipping configuration with PREFILL_CHUNK=8192: exactness diff, the sweep (8K 32K 128K), the stall probe
# The server must be running with the configuration of MODE.
set -u
MODE=$1
R=$HOME/projects/inkling2-mini-k-prefill
cd "$R" || exit 1
URL=http://$(sed -n 's/^NODES=//p' .env):8000
P=bench/records/2026-11-23-prefill-push-exact; S=bench/records/2026-11-23-prefill-push
mkdir -p "$P/raw/$MODE" "$S/raw/$MODE"
M=inkling-2-mini-1x-dgx-spark
./inkling2mini.sh memwatch begin > "$P/raw/$MODE/memwatch-begin.txt" 2>&1
sweep() { # sizes cache?
  env PREFILL_SIZES=$1 ${2:+PREFILL_CACHE_DEFAULT=1} ./inkling2mini.sh bench prefill-sweep > "$S/raw/$MODE/sweep.out" 2>&1
  cp -r "$(ls -d bench/runs/*-prefill-sweep | tail -1)" "$S/raw/$MODE/sweep"
}
case $MODE in
  ship)
    python3 bench/tools/exactness.py record "$P/raw/ship/exactness.json" --max-tokens 128 --label ship --url "$URL" > "$P/raw/ship/exactness-record.out" 2>&1
    python3 bench/tools/exactness.py diff bench/records/2026-11-23-prefill-exact/raw/base-7da8c61/exactness.json "$P/raw/ship/exactness.json" > "$P/raw/ship/exactness-diff.out" 2>&1; echo "exit $?" >> "$P/raw/ship/exactness-diff.out"
    python3 bench/tools/bench.py --url "$URL" --model $M --max-model-len 262144 --phases warm,decode,conc,prefill-32k --ref bench/records/2026-11-23-prefill-exact/raw/pinned-replies-engine-51f2993.json --out "$P/raw/ship/bench-ref.json" > "$P/raw/ship/bench.out" 2> "$P/raw/ship/bench.err"; echo "exit $?" >> "$P/raw/ship/bench.out"
    python3 bench/tools/specedge.py --url "$URL" --out "$P/raw/ship/specedge.json" > "$P/raw/ship/specedge.out" 2>&1; echo "exit $?" >> "$P/raw/ship/specedge.out"
    python3 bench/tools/exactness.py concurrent --max-tokens 64 --url "$URL" > "$P/raw/ship/concurrent.out" 2>&1; echo "exit $?" >> "$P/raw/ship/concurrent.out"
    sweep 8192,32768,131072,229376 cache
    python3 bench/tools/stall_probe.py --url "$URL" --prompt-tokens 32768 --out "$S/raw/ship/stall-4096.json" > "$S/raw/ship/stall-4096.out" 2>&1 ;;
  nospec) sweep 8192,32768,131072,229376 ;;
  c8192)
    python3 bench/tools/exactness.py record "$P/raw/c8192/exactness.json" --max-tokens 128 --label c8192 --url "$URL" > "$P/raw/c8192/exactness-record.out" 2>&1
    python3 bench/tools/exactness.py diff bench/records/2026-11-23-prefill-exact/raw/base-7da8c61/exactness.json "$P/raw/c8192/exactness.json" > "$P/raw/c8192/exactness-diff.out" 2>&1; echo "exit $?" >> "$P/raw/c8192/exactness-diff.out"
    sweep 8192,32768,131072 cache
    python3 bench/tools/stall_probe.py --url "$URL" --prompt-tokens 32768 --out "$S/raw/c8192/stall-8192.json" > "$S/raw/c8192/stall-8192.out" 2>&1 ;;
esac
./inkling2mini.sh memwatch report > "$S/raw/$MODE/memwatch-report.txt" 2>&1
echo done > "$S/raw/$MODE/done.txt"
