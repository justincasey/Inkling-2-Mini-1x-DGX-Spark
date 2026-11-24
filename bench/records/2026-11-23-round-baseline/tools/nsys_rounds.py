#!/usr/bin/env python3
"""nsys_rounds.py REPORT.sqlite [MAXROUNDS]: the GPU/host timeline of drafted decode rounds from an nsys export of the server.

A round = the interval between two window-graph launches (cudaGraphLaunch of the engine thread). The window's own activities are the
ones whose correlationId is that launch's (graph nodes); the rest between two windows is the drafter (proposal, feed) and the commit.
Prints, averaged over rounds with a full set: window span (first node start .. last node end), window busy (union of node intervals),
the drafter's span and busy, the commit's, the idle gaps by place (window end -> first drafter/commit activity: host reads the candidates,
accepts, launches the commit; drafter end -> next window start: host reads the proposal, plans, uploads, launches), the launch latency
(cudaGraphLaunch start -> first node start), kernels per round, and the top kernels of the window and of the drafter.
"""
import sqlite3, sys, statistics, collections

db = sqlite3.connect(sys.argv[1])
maxr = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
S = dict(db.execute("select id,value from StringIds"))
k = db.execute("select start,end,correlationId,shortName,streamId,graphNodeId from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
m = db.execute("select start,end,correlationId,bytes,copyKind from CUPTI_ACTIVITY_KIND_MEMCPY").fetchall()
ms = db.execute("select start,end,correlationId from CUPTI_ACTIVITY_KIND_MEMSET").fetchall()
rt = db.execute("select start,end,correlationId,nameId,globalTid from CUPTI_ACTIVITY_KIND_RUNTIME").fetchall()
name = lambda i: S.get(i, "?")
glaunch = [r for r in rt if name(r[3]).startswith("cudaGraphLaunch")]
tids = collections.Counter(r[4] for r in glaunch)
tid = tids.most_common(1)[0][0]
glaunch = sorted((r for r in glaunch if r[4] == tid), key=lambda r: r[0])
print("graph launches on the engine thread:", len(glaunch), "kernels", len(k), "memcpy", len(m), "memset", len(ms))
acts = [(a[0], a[1], a[2], name(a[3]), "k") for a in k] + [(a[0], a[1], a[2], "memcpy%d_%dB" % (a[4], a[3]), "c") for a in m] + \
       [(a[0], a[1], a[2], "memset", "s") for a in ms]
acts.sort()
wcorr = {r[2]: i for i, r in enumerate(glaunch)}
win = collections.defaultdict(list)
oth = []
for a in acts:
    if a[2] in wcorr:
        win[wcorr[a[2]]].append(a)
    else:
        oth.append(a)

def union(iv):
    iv = sorted(iv)
    tot, cur = 0, None
    for s, e in iv:
        if cur is None or s > cur[1]:
            if cur: tot += cur[1] - cur[0]
            cur = [s, e]
        else:
            cur[1] = max(cur[1], e)
    if cur: tot += cur[1] - cur[0]
    return tot

# the sync/api calls of the engine thread between windows
def api_in(t0, t1):
    out = collections.defaultdict(float)
    for r in rt:
        if r[4] == tid and r[0] >= t0 and r[1] <= t1:
            out[name(r[3])] += (r[1] - r[0]) / 1e6
    return out

rows = []
for i in range(1, min(len(glaunch) - 1, maxr + 1)):
    w, wn = win.get(i), win.get(i + 1)
    if not w or not wn:
        continue
    ws, we = min(a[0] for a in w), max(a[1] for a in w)
    nws = min(a[0] for a in wn)
    mid = [a for a in oth if we <= a[0] < nws]
    if not mid:
        continue
    ds, de = min(a[0] for a in mid), max(a[1] for a in mid)
    # the drafter's kernels (gb_i2d_*, exl3 at b*8 rows) cannot be told from the commit's by name alone: split at the longest idle gap in mid
    mid.sort()
    gaps = []
    end = mid[0][1]
    for a in mid[1:]:
        gaps.append((a[0] - end, end, a[0]))
        end = max(end, a[1])
    big = max(gaps) if gaps else (0, de, de)
    rows.append(dict(
        win_span=(we - ws) / 1e6, win_busy=union([(a[0], a[1]) for a in w]) / 1e6, win_n=len(w),
        launch_lat=(ws - glaunch[i][0]) / 1e6,
        post_gap=(ds - we) / 1e6, mid_span=(de - ds) / 1e6, mid_busy=union([(a[0], a[1]) for a in mid]) / 1e6, mid_n=len(mid),
        mid_biggap=big[0] / 1e6, pre_gap=(nws - de) / 1e6, period=(nws - ws) / 1e6,
        win_i=i, ws=ws, we=we, nws=nws, de=de, ds=ds))
print("rounds analysed:", len(rows))
def med(key):
    return statistics.median(r[key] for r in rows)
for key in ("period", "win_span", "win_busy", "win_n", "launch_lat", "post_gap", "mid_span", "mid_busy", "mid_n", "mid_biggap", "pre_gap"):
    print(f"{key:12s} median {med(key):10.3f}  mean {statistics.fmean(r[key] for r in rows):10.3f}")
# bucket by window kernel count (rows of the window): 1 row windows have fewer nodes? print period per win_span quartile
rows.sort(key=lambda r: r["win_span"])
for lo, hi in ((0, len(rows) // 3), (len(rows) // 3, 2 * len(rows) // 3), (2 * len(rows) // 3, len(rows))):
    sub = rows[lo:hi]
    if sub:
        print("win_span %.1f..%.1f ms: n=%d period %.2f win_busy %.2f mid_span %.2f mid_busy %.2f post_gap %.3f pre_gap %.3f" % (
            sub[0]["win_span"], sub[-1]["win_span"], len(sub), statistics.fmean(r["period"] for r in sub), statistics.fmean(r["win_busy"] for r in sub),
            statistics.fmean(r["mid_span"] for r in sub), statistics.fmean(r["mid_busy"] for r in sub), statistics.fmean(r["post_gap"] for r in sub),
            statistics.fmean(r["pre_gap"] for r in sub)))
# kernel tables: window and mid, per round
nr = len(rows)
tw, tm = collections.Counter(), collections.Counter()
cw, cm = collections.Counter(), collections.Counter()
for r in rows:
    for a in win[r["win_i"]]:
        tw[a[3]] += a[1] - a[0]; cw[a[3]] += 1
    for a in oth:
        if r["we"] <= a[0] < r["nws"]:
            tm[a[3]] += a[1] - a[0]; cm[a[3]] += 1
for title, t, c in (("WINDOW kernels (per round)", tw, cw), ("BETWEEN-WINDOWS kernels (drafter, commit; per round)", tm, cm)):
    print(title, "total ms/round %.3f" % (sum(t.values()) / nr / 1e6))
    for n, v in t.most_common(14):
        print(f"  {v / nr / 1e6:8.3f} ms  x{c[n] / nr:7.1f}  avg {v / c[n] / 1e3:8.1f} us  {n[:90]}")
# host API calls between windows
tot = collections.defaultdict(float)
for r in rows[:200]:
    for n, v in api_in(r["we"] - 3_000_000, r["nws"] + 200_000).items():
        tot[n] += v
print("engine-thread CUDA API ms per round (windows [we-3ms, next ws+0.2ms]):")
for n, v in sorted(tot.items(), key=lambda x: -x[1])[:12]:
    print(f"  {v / min(len(rows), 200):8.3f}  {n}")
