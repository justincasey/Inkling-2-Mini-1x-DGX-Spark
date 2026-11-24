#!/usr/bin/env bash
O=/tmp/fuse/fin2; mkdir -p $O; cd $HOME/projects/inkling2-mini-k-fuse
./inkling2mini.sh start > $O/start.log 2>&1 || { tail -5 $O/start.log; exit 1; }
python3 bench/tools/gates.py spec $O/gates-spec.json > $O/gates-spec.out 2>&1
python3 bench/tools/gates.py cache $O/gates-cache.json > $O/gates-cache.out 2>&1
./inkling2mini.sh stop > $O/stop.log 2>&1
SPEC=none PREFIX_CACHE=off VISION=off ./inkling2mini.sh start > $O/start-serial.log 2>&1 || { tail -5 $O/start-serial.log; exit 1; }
./inkling2mini.sh bench decode > $O/serial-warm.log 2>&1
python3 bench/rexl3/sweep_decode.py --url http://node-2:8000 --model inkling-2-mini-1x-dgx-spark --window 262144 --max-total 1048576 --phase both --ctxs 4096 --conc 1,2,4 --modes serial --depth-modes serial --reps 3 --depth-reps 3 --out $O/sweep.json > $O/sweep.log 2>&1
./inkling2mini.sh bench decode > $O/serial-bench.log 2>&1
./inkling2mini.sh stop > $O/stop2.log 2>&1
echo done
