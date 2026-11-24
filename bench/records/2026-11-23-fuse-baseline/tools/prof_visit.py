#!/usr/bin/env python3
"""prof_visit.py DB.sqlite [STEPS] [timeline VISIT_INDEX]: per-kernel table of the replayed decode windows of an `nsys_run.sh` profile.

A step is the span from one unembed launch (`linear_kernel` with gridX >= 1000) to the next (so it holds the eager commit launches between two
windows, as the old prof_an.py did). Per kernel (name + launch geometry): launches a step, median microseconds, microseconds a step summed over
lanes (side streams overlap, so the sum exceeds the span), share of the span, and for the dense EXL3 and bf16 linears the bytes the launch must
read and the bandwidth it reached. `timeline N` prints window N-from-last's kernels with their start offsets and lane numbers."""
import sqlite3, sys, collections, statistics, re

db = sys.argv[1]
steps = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 20
c = sqlite3.connect(db)
rows = c.execute("select k.start,k.end,s.value,k.gridX,k.gridY,k.gridZ,k.blockX,k.streamId,k.graphNodeId from CUPTI_ACTIVITY_KIND_KERNEL k "
                 "join StringIds s on s.id=k.demangledName order by k.start").fetchall()
marks = [i for i, r in enumerate(rows) if 'linear_kernel' in r[2] and r[3] >= 1000 and r[8]]
m = marks[-(steps + 1):]
sel = rows[m[0]:m[-1]]
span = [(rows[m[i + 1]][0] - rows[m[i]][0]) for i in range(len(m) - 1)]
mspan = statistics.median(span)


def short(n):
    n = n.replace('void <unnamed>::', '').replace('<unnamed>::', '').replace('(int)', '').replace('(bool)', '')
    n = re.sub(r'\(.*$', '', n)
    return n


if len(sys.argv) > 3 and sys.argv[3] == 'timeline':
    k = int(sys.argv[4])
    a, b = m[-1 - k - 1], m[-1 - k]
    t0 = rows[a][0]
    lanes = {}
    for r in rows[a:b]:
        ln = lanes.setdefault(r[7], len(lanes))
        print("%9.1f %7.1f L%d g(%d,%d,%d)x%d %s" % ((r[0] - t0) / 1e3, (r[1] - r[0]) / 1e3, ln, r[3], r[4], r[5], r[6], short(r[2])))
    sys.exit()

agg = collections.defaultdict(list)
for r in sel:
    agg[(short(r[2]), r[3], r[4], r[5], r[6])].append(r[1] - r[0])
print("steps %d  kernels/step %.1f  span median %.0f us (min %.0f)" % (steps, len(sel) / steps, mspan / 1e3, min(span) / 1e3))
busy = sum(r[1] - r[0] for r in sel) / steps
print("sum of kernel durations/step %.0f us (side streams overlap)" % (busy / 1e3))
print("%-52s %-18s %7s %8s %9s %6s" % ("kernel", "grid x block", "n/step", "med us", "us/step", "%span"))
tot = 0
for k, v in sorted(agg.items(), key=lambda x: -sum(x[1]))[:int(sys.argv[5]) if len(sys.argv) > 5 else 40]:
    us = sum(v) / steps / 1e3
    tot += us
    print("%-52s %-18s %7.1f %8.1f %9.1f %5.1f%%" % (k[0][:52], "%dx%dx%d x%d" % k[1:], len(v) / steps, statistics.median(v) / 1e3, us, 100 * us * 1e3 / mspan))
print("listed total %.0f us/step" % tot)
