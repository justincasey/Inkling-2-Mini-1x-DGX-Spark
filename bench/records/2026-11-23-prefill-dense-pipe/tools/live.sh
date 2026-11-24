#!/usr/bin/env bash
# live.sh MODE LABEL: the live checks and cold-prefill sweeps of the dense pipe on node-3, from the recipe worktree (the server must already be running with the
# configuration of MODE; the node address is in the worktree's git-ignored .env).
#   ship    shipping configuration (the .env defaults: SPEC=dspark PREFIX_CACHE=on VISION=on, chunk 4,096): exactness record + diff against the 7da8c61 record of
#           `2026-11-23-prefill-exact`, bench.py --ref PASS pin, specedge.py, exactness concurrent
#   sweep   the cold-prefill sweep (8K 32K 128K 224K, unique prompts, one warm-up pass discarded, medians of three kept passes) into raw/LABEL/
# LABEL names the configuration (ship-pipe-on, ship-pipe-off, nospec-pipe-on, nospec-pipe-off).
set -u
MODE=$1; LABEL=$2
cd "$(dirname "$0")/../../../.." || exit 1
URL=http://$(sed -n 's/^NODES=//p' .env):8000
S=bench/records/2026-11-23-prefill-dense-pipe; P=bench/records/2026-11-23-prefill-dense-pipe-exact
mkdir -p "$S/raw/$LABEL" "$P/raw/$LABEL"
M=inkling-2-mini-1x-dgx-spark
case $MODE in
  ship)
    ./inkling2mini.sh memwatch begin > "$P/raw/$LABEL/memwatch-begin.txt" 2>&1
    python3 bench/tools/exactness.py record "$P/raw/$LABEL/exactness.json" --max-tokens 128 --label "$LABEL" --url "$URL" > "$P/raw/$LABEL/exactness-record.out" 2>&1
    python3 bench/tools/exactness.py diff bench/records/2026-11-23-prefill-exact/raw/base-7da8c61/exactness.json "$P/raw/$LABEL/exactness.json" > "$P/raw/$LABEL/exactness-diff.out" 2>&1; echo "exit $?" >> "$P/raw/$LABEL/exactness-diff.out"
    python3 bench/tools/bench.py --url "$URL" --model $M --max-model-len 262144 --phases warm,decode,conc,prefill-32k --ref bench/records/2026-11-23-prefill-exact/raw/pinned-replies-engine-51f2993.json --out "$P/raw/$LABEL/bench-ref.json" > "$P/raw/$LABEL/bench.out" 2> "$P/raw/$LABEL/bench.err"; echo "exit $?" >> "$P/raw/$LABEL/bench.out"
    python3 bench/tools/specedge.py --url "$URL" --out "$P/raw/$LABEL/specedge.json" > "$P/raw/$LABEL/specedge.out" 2>&1; echo "exit $?" >> "$P/raw/$LABEL/specedge.out"
    python3 bench/tools/exactness.py concurrent --max-tokens 64 --url "$URL" > "$P/raw/$LABEL/concurrent.out" 2>&1; echo "exit $?" >> "$P/raw/$LABEL/concurrent.out"
    ./inkling2mini.sh memwatch report > "$P/raw/$LABEL/memwatch-report.txt" 2>&1 ;;
  sweep)
    env PREFILL_SIZES=${SIZES:-8192,32768,131072,229376} ${CACHE:+PREFILL_CACHE_DEFAULT=1} ./inkling2mini.sh bench prefill-sweep > "$S/raw/$LABEL/sweep.out" 2>&1
    cp -r "$(ls -d bench/runs/*-prefill-sweep | tail -1)" "$S/raw/$LABEL/sweep" ;;
esac
echo done > "$S/raw/$LABEL/done-$MODE.txt"
