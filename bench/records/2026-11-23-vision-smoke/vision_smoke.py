#!/usr/bin/env python3
"""vision-smoke: image requests against the Mini's server (stdlib only; images from bench/tools/synth_images.py).

usage: vision_smoke.py URL OUTDIR   (URL like http://HOST:8000/v1; raw requests and replies go to OUTDIR/*.json)
Checks (each prints PASS/FAIL, exit 1 on any FAIL):
  describe-<name>   each synthetic image alone: the reply names its colours
  repeat-<name>     the same request again: identical generated token ids
  three-images      one request with all three images: the reply names all four colours
  concurrent        two image requests and a text request at once: each equals its own solo reply (token ids)
  cap-image         an image of exactly max-image-tokens (4,096) cells: served, crossing a prefill chunk, the reply is non-empty
  over-cap          an image of 4,624 tokens: refused by name with HTTP 400
"""
import json, os, sys, threading, time, urllib.error, urllib.request
sys.path.insert(0, os.environ.get("TOOLS", "bench/tools"))
import synth_images as si

URL, OUT = sys.argv[1], sys.argv[2]
M = "inkling-2-mini-1x-dgx-spark"
os.makedirs(OUT, exist_ok=True)
results = []

def post(body, name):
    req = urllib.request.Request(f"{URL}/chat/completions", json.dumps({"model": M, "temperature": 0, "return_token_ids": True, **body}).encode(), {"content-type": "application/json"})
    t = time.time()
    try:
        r = urllib.request.urlopen(req, timeout=1800); status, raw = r.status, r.read()
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read()
    dt = time.time() - t
    d = json.loads(raw)
    slim = json.loads(json.dumps(d))
    json.dump({"request": {k: (v if k != "messages" else "<messages: %d, image data urls elided>" % len(v)) for k, v in body.items()}, "status": status, "wall_s": round(dt, 2), "reply": slim}, open(f"{OUT}/{name}.json", "w"), indent=1)
    return status, d, dt

def text_of(d):
    m = d["choices"][0]["message"]
    return ((m.get("reasoning_content") or "") + " " + (m.get("content") or "")).lower()

def ids_of(d):
    c = d["choices"][0]
    return c.get("token_ids") or d.get("token_ids") or c.get("message", {}).get("token_ids")

def img_msg(question, urls):
    return [{"role": "user", "content": [{"type": "text", "text": question}] + [{"type": "image_url", "image_url": {"url": u}} for u in urls]}]

def report(name, ok, line):
    results.append((name, ok)); print(("PASS " if ok else "FAIL ") + name + ": " + line, flush=True)

urls = {n: si.data_url(si.image_png(n)) for n in si.IMAGES}
solo = {}
for n, (_, q, words) in si.IMAGES.items():
    body = {"messages": img_msg(q, [urls[n]]), "reasoning_effort": "none", "max_tokens": 200}
    s, d, dt = post(body, f"describe-{n}")
    ok = s == 200 and all(w in text_of(d) for w in words)
    solo[n] = (body, ids_of(d) if s == 200 else None)
    report(f"describe-{n}", ok, f"{dt:.1f}s, usage {d.get('usage', {}).get('prompt_tokens')} prompt tokens; reply: {text_of(d)[:160]!r}")
    s2, d2, _ = post(body, f"repeat-{n}")
    report(f"repeat-{n}", s2 == 200 and ids_of(d2) == ids_of(d) and ids_of(d) is not None, f"{len(ids_of(d) or [])} ids, identical={ids_of(d2) == ids_of(d)}")

body = {"messages": img_msg("These are three images. Name the main colour of each. Answer briefly.", list(urls.values())), "reasoning_effort": "none", "max_tokens": 300}
s, d, dt = post(body, "three-images")
report("three-images", s == 200 and all(w in text_of(d) for w in ["red", "blue", "green", "yellow"]), f"{dt:.1f}s, {d.get('usage', {}).get('prompt_tokens')} prompt tokens; reply: {text_of(d)[:200]!r}")

text_body = {"messages": [{"role": "user", "content": "Write two sentences about the sea."}], "reasoning_effort": "none", "max_tokens": 120}
s, dtx, _ = post(text_body, "text-solo")
text_solo = ids_of(dtx)
outs = {}
def go(k, b):
    outs[k] = post(b, f"concurrent-{k}")
th = [threading.Thread(target=go, args=a) for a in [("red", solo["red-square"][0]), ("blue", solo["blue-circle"][0]), ("text", text_body)]]
[t.start() for t in th]; [t.join() for t in th]
ok = outs["red"][1] and ids_of(outs["red"][1]) == solo["red-square"][1] and ids_of(outs["blue"][1]) == solo["blue-circle"][1] and ids_of(outs["text"][1]) == text_solo and text_solo is not None
report("concurrent", bool(ok), "red==solo %s, blue==solo %s, text==solo %s" % (ids_of(outs["red"][1]) == solo["red-square"][1], ids_of(outs["blue"][1]) == solo["blue-circle"][1], ids_of(outs["text"][1]) == text_solo))

def big(cells):
    px = cells * 40
    row = bytes(si.WHITE) * px
    import struct, zlib
    raw = b"".join(b"\x00" + (bytes(si.RED) * px if (y // 40) % 2 == 0 else row) for y in range(px))
    def chunk(k, data): return struct.pack(">I", len(data)) + k + data + struct.pack(">I", zlib.crc32(k + data) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", px, px, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 1)) + chunk(b"IEND", b"")

s, d, dt = post({"messages": img_msg("Describe the stripes in one sentence.", [si.data_url(big(64))]), "reasoning_effort": "none", "max_tokens": 60}, "cap-image")
report("cap-image", s == 200 and bool(text_of(d).strip()), f"status {s}, {dt:.1f}s, {d.get('usage', {}).get('prompt_tokens')} prompt tokens; reply: {text_of(d)[:160]!r}")
s, d, dt = post({"messages": img_msg("Describe it.", [si.data_url(big(68))]), "reasoning_effort": "none", "max_tokens": 20}, "over-cap")
msg = json.dumps(d)
report("over-cap", s == 400 and "token" in msg.lower(), f"status {s}: {msg[:300]}")
bad = [n for n, ok in results if not ok]
print("RESULT", "pass" if not bad else "FAIL " + ",".join(bad))
sys.exit(1 if bad else 0)
