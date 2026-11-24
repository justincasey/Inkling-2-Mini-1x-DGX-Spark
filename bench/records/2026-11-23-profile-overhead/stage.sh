#!/bin/bash
# usage: stage.sh LABEL  (env passes through to the controller); fine-grained node sampler during boot, then decode phase, then sampler again
L=$1; OUT=/tmp/kprof/$L; mkdir -p $OUT
cd $HOME/projects/inkling2-mini-k-profile
ssh -o BatchMode=yes spark@node-4 'for i in $(seq 1 120); do echo "$(date +%s.%N | cut -c1-14) gpu_mib=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | awk "{s+=\$1} END {print s+0}") avail_mib=$(awk "/^MemAvailable:/ {printf \"%d\", \$2/1024}" /proc/meminfo)"; sleep 0.2; done' > $OUT/boot-samples.txt 2>&1 &
S=$!
./inkling2mini.sh start > $OUT/start.txt 2>&1; tail -4 $OUT/start.txt | cut -c1-600
wait $S
ssh -o BatchMode=yes spark@node-4 'grep -E "event=(boot|inkling2_loop_plan|memory_view|graph|overlap)|graphs|overlap" /home/spark/inkling-2-mini-1x-dgx-spark/logs/serve.log | cut -c1-700' > $OUT/boot-events.txt
