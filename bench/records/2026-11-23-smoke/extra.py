#!/usr/bin/env python3
"""Beyond the five smoke requests: one long prompt (prefill rate across chunks) and four concurrent requests (4 slots)."""
import json, sys, time, urllib.request, threading
URL, OUT, M = sys.argv[1], sys.argv[2], "inkling-2-mini-1x-dgx-spark"
def post(body):
    t = time.time()
    r = urllib.request.urlopen(urllib.request.Request(f"{URL}/completions", json.dumps({"model": M, "temperature": 0, **body}).encode(), {"content-type": "application/json"}), timeout=1800)
    d = json.loads(r.read()); return d, time.time() - t
res = {}
para = "The quick brown fox jumps over the lazy dog while the river runs past the old stone mill. "
d, dt = post({"prompt": para * 600 + "\nSummary of the text above in one sentence:", "max_tokens": 32})
res["long-prompt"] = {"wall_s": round(dt, 2), "usage": d["usage"], "text": d["choices"][0]["text"]}
print("long", dt, d["usage"], repr(d["choices"][0]["text"])[:200])
d1, dt1 = post({"prompt": "Once upon a time", "max_tokens": 64})
outs = {}
def go(i):
    d, dt = post({"prompt": "Once upon a time", "max_tokens": 64}); outs[i] = (d, dt)
ts = [threading.Thread(target=go, args=(i,)) for i in range(4)]
t = time.time(); [x.start() for x in ts]; [x.join() for x in ts]; wall = time.time() - t
same = len({outs[i][0]["choices"][0]["text"] for i in range(4)}) == 1 and outs[0][0]["choices"][0]["text"] == d1["choices"][0]["text"]
res["concurrent-4"] = {"solo_wall_s": round(dt1, 2), "wall_s": round(wall, 2), "all_equal_to_solo": same, "text": d1["choices"][0]["text"]}
print("solo", dt1, "4 concurrent", wall, "equal to solo:", same)
json.dump(res, open(f"{OUT}/extra.json", "w"), indent=1)
