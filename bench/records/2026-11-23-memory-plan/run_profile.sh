#!/usr/bin/env bash
# run_profile.sh LABEL [VAR=value ...]: start the server through the controller with the given settings, send a few requests (the second
# sighting of a window shape captures its graph), save the facts, stop. Output in /tmp/mp/LABEL/.
set -u
LABEL=$1; shift
cd $HOME/projects/inkling2-mini-k-boot
OUT=/tmp/mp/$LABEL; mkdir -p $OUT
env "$@" ./inkling2mini.sh print-serve > $OUT/serve-cmd.txt 2>&1
env "$@" ./inkling2mini.sh start > $OUT/start.txt 2>&1; echo "start rc=$?" >> $OUT/start.txt
if grep -q "start: ready" $OUT/start.txt; then
  ./inkling2mini.sh status > $OUT/status-ready.txt 2>&1
  URL=http://node-1:8000/v1
  for i in 1 2 3 4; do curl -s -m 120 $URL/chat/completions -H 'content-type: application/json' -d '{"model":"inkling-2-mini-1x-dgx-spark","messages":[{"role":"user","content":"Say OK."}],"max_tokens":24,"temperature":0}' > /dev/null; done
  curl -s -m 600 $URL/completions -H 'content-type: application/json' -d '{"model":"inkling-2-mini-1x-dgx-spark","prompt":"Once upon a time","max_tokens":96,"temperature":0}' > /dev/null
  ./inkling2mini.sh status > $OUT/status-after-traffic.txt 2>&1
fi
./inkling2mini.sh memwatch report > $OUT/memwatch.txt 2>&1
ssh -o BatchMode=yes spark@node-1 'cat /home/spark/inkling-2-mini-1x-dgx-spark/logs/serve.log; echo ---; cat /home/spark/inkling-2-mini-1x-dgx-spark/logs/memguard.min 2>/dev/null; echo ---; grep -E "MemTotal|MemAvailable" /proc/meminfo; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader' > $OUT/serve-and-mem.txt 2>&1
./inkling2mini.sh stop > $OUT/stop.txt 2>&1
grep -E "inkling2_loop_plan|mem_stage|graph_captured|prefix_cache_plan|weights_differ|level=(error|fatal)" $OUT/serve-and-mem.txt | cut -c1-420 > $OUT/events.txt
