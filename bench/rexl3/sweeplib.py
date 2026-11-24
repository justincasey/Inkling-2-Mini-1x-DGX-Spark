"""Shared pieces of the performance-profile sweeps (sweep_decode.py, sweep_prefill.py; stdlib only, importable without a server).

The Mini runs on ONE Spark (TP1), so a sweep is one server process: a URL from `inkling_api.add_api_args`, no ranks.
Everything that measures reuses `bench/tools/bench.py` (solo, together, hist, long_ids, long_text, the server's histograms
`rexl3_request_ttft_seconds` / `rexl3_request_e2e_seconds`) and `bench/tools/inkling_api.py`; nothing here is a second client.

* GRID / DECODE_CONC / PREFILL_GRID   the documented cells of a profile; `grid()` truncates them at the window
* ModeApi        an Api wrapper: memoised /tokenize (a 1M-token prompt is tokenized once, not once per cell), and the serial mode
                 (`"draft": false`, no speculation) injected into every completion
* lane_prompt    the prompt of exactly-or-about n tokens for lane k: lane 0 is bench.long_ids / bench.long_text, other lanes
                 the same stream from a shifted block seed, so concurrent streams differ (identical prompts would make
                 every stream pick the same experts, which flatters a MoE's concurrent decode)
* solo_cached    one request WITH the prefix cache, its TTFT and decode speed from the server's histograms (bench.solo is cold only)
* stream_one / fire_streams   a streamed /v1/completions with the arrival stamps of every event, started behind a barrier
* overlap_metrics   the decode of c streams over their OVERLAP WINDOW (from the last first token to the first last token)
"""

from __future__ import annotations

import contextlib
import os
import statistics
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(os.path.dirname(HERE), "tools")
sys.path.insert(0, TOOLS)
import bench  # noqa: E402
import inkling_api as ia  # noqa: E402

K = 1024
# The profile's context grid (prompt tokens): 4K ... 1M. A cell runs only when its prompt plus the reply fits the window.
DECODE_GRID = (4 * K, 16 * K, 32 * K, 64 * K, 128 * K, 256 * K, 512 * K, 1024 * K)
PREFILL_GRID = (8 * K, 16 * K, 32 * K, 64 * K, 128 * K, 256 * K, 512 * K, 1024 * K)
DECODE_CONC = (1, 2, 4, 8)           # streams
PREFILL_CONC = (1, 2, 4)
WINDOWS = (262_144, 1_048_576)       # the Mini's profiles: bootstrap and long (max)
DEFAULT_WINDOW = 262_144
RESERVE = 4096                       # window tokens kept free under a prompt for the reply (and the template)
DEPTH_TOKENS = 256                   # generated per stream in a depth cell
IDS_LIMIT = 131_072                  # above this a prompt goes as text (the server refuses > 300,000 JSON nodes in a body)
MAX_TOTAL_TOKENS = 1_048_576         # cap on streams x prompt tokens of one concurrent prefill / depth cell
LANE_STRIDE = 100_000                # block seed distance between lanes


def parse_list(s: str, window: int, reserve: int = RESERVE) -> list[int]:
    """'4096,32768,top' -> ints; `top` is window - reserve. Order kept, duplicates dropped."""
    out: list[int] = []
    for tok in (t.strip() for t in s.split(",") if t.strip()):
        v = window - reserve if tok == "top" else int(tok)
        if v not in out:
            out.append(v)
    return out


def grid(values, window: int, reserve: int = RESERVE) -> list[int]:
    """The cells of `values` that fit: prompt + reserve <= window."""
    return [v for v in values if v + reserve <= window]


def stats(v) -> dict:
    v = [x for x in v if x is not None]
    if not v:
        return {"median": None, "min": None, "max": None, "n": 0}
    return {"median": statistics.median(v), "min": min(v), "max": max(v), "n": len(v)}


def fits(c: int, tokens: int, max_total: int) -> bool:
    return c * tokens <= max_total


class ModeApi:
    """The Api every sweep drives. `serial`: every completion carries `"draft": false` (the reference path without speculation)."""

    def __init__(self, api: ia.Api, serial: bool = False):
        self._api, self.serial, self._tok = api, serial, {}

    def __getattr__(self, name):
        return getattr(self._api, name)

    def tokenize(self, text: str) -> list[int]:
        got = self._tok.get(text)
        if got is None:
            got = self._tok[text] = self._api.tokenize(text)
        return got

    def complete(self, prompt, max_tokens: int, **extra):
        if self.serial:
            extra = {"draft": False, **extra}
        return self._api.complete(prompt, max_tokens, **extra)

    def with_mode(self, serial: bool) -> "ModeApi":
        other = ModeApi(self._api, serial)
        other._tok = self._tok
        return other


@contextlib.contextmanager
def lane_blocks(lane: int):
    """bench.block_text with the seed shifted by the lane: lane 0 is the bench's own stream."""
    orig = bench.block_text
    if lane:
        bench.block_text = lambda i, o=orig, s=lane * LANE_STRIDE: o(s + i)
    try:
        yield
    finally:
        bench.block_text = orig


def lane_prompt(api, n: int, lane: int = 0):
    """A prompt of about n tokens (exact by construction as ids up to IDS_LIMIT; text above, its real size is usage.prompt_tokens)."""
    with lane_blocks(lane):
        return bench.long_ids(api, n) if n <= IDS_LIMIT else bench.long_text(api, n)


def solo_cached(api, prompt, max_tokens: int) -> dict:
    """One request with the prefix cache on, alone: the bench.solo record (TTFT and decode speed from the server's histograms)
    plus cached_tokens. Whether it hit is for the caller to judge from cached_tokens."""
    m0 = api.metrics()
    t0, e0 = bench.hist(m0, bench.TTFT), bench.hist(m0, bench.E2E)
    resp, wall = api.complete(prompt, max_tokens, prefix_cache=True)
    m1 = api.metrics()
    t1, e1 = bench.hist(m1, bench.TTFT), bench.hist(m1, bench.E2E)
    r = bench.reply(prompt, resp, wall)
    ttft, e2e = t1[0] - t0[0], e1[0] - e0[0]
    r["ttft_s"] = round(ttft, 4)
    r["decode_s"] = round(e2e - ttft, 4)
    r["decode_tok_s"] = round((r["completion_tokens"] - 1) / (e2e - ttft), 3) if e2e > ttft and r["completion_tokens"] > 1 else None
    return r


def slim(r: dict) -> dict:
    """A bench reply without the id list (a sweep file keeps numbers, not 256 ids a cell)."""
    out = {k: v for k, v in r.items() if k not in ("generated_ids", "prompt_sha256")}
    out["generated_tokens"] = len(r.get("generated_ids") or [])
    return out


# ── streamed requests ──
def stream_one(api, body: dict, barrier: threading.Barrier | None = None) -> dict:
    """A streamed /v1/completions. Events are (monotonic arrival stamp, ids in the event); the token count is usage.completion_tokens."""
    body = dict(body, stream_options={"include_usage": True}, return_token_ids=True)
    out = {"events": [], "usage": None, "error": None}
    if barrier is not None:
        barrier.wait()
    t0 = time.monotonic()
    out["t0"] = t0
    try:
        for dt, ev in api.stream("/v1/completions", body):
            if "error" in ev:
                out["error"] = ev["error"]
            for ch in ev.get("choices") or []:
                ids = ch.get("token_ids") or []
                if ids or ch.get("text"):
                    out["events"].append((t0 + dt, max(1, len(ids))))
            if ev.get("usage"):
                out["usage"] = ev["usage"]
    except Exception as e:  # noqa: BLE001 - recorded in the cell
        out["error"] = f"{type(e).__name__}: {e}"
    out["tokens"] = int((out["usage"] or {}).get("completion_tokens") or 0)
    return out


def fire_streams(api, bodies: list[dict]) -> list[dict]:
    barrier = threading.Barrier(len(bodies)) if len(bodies) > 1 else None
    res: list = [None] * len(bodies)

    def one(i):
        res[i] = stream_one(api, bodies[i], barrier)
    th = [threading.Thread(target=one, args=(i,)) for i in range(len(bodies))]
    for t in th:
        t.start()
    for t in th:
        t.join()
    return res


def stream_rate(s: dict) -> float | None:
    """(usage tokens - 1) / (last - first event): the stream's decode speed."""
    ev = s["events"]
    if s.get("error") or len(ev) < 2 or s["tokens"] < 2:
        return None
    span = ev[-1][0] - ev[0][0]
    return (s["tokens"] - 1) / span if span > 0 else None


def overlap_metrics(streams: list[dict], min_s: float = 3.0, min_frac: float = 0.8) -> dict:
    """Decode of concurrent streams over their overlap window [max(first), min(last)]. Tokens of a stream inside the window =
    its usage.completion_tokens x the share of its event tokens that arrived in the window (the count is the server's, the
    chunks only place it in time). aggregate = window tokens / window; ok when the window is >= min_s and >= min_frac of the
    mean stream span."""
    if not streams or any(s.get("error") or len(s["events"]) < 2 or s["tokens"] < 2 for s in streams):
        return {"ok": False, "reason": "a stream failed or produced fewer than 2 events"}
    w0, w1 = max(s["events"][0][0] for s in streams), min(s["events"][-1][0] for s in streams)
    spans = [s["events"][-1][0] - s["events"][0][0] for s in streams]
    mean_span = sum(spans) / len(spans)
    overlap = w1 - w0
    win = 0.0
    for s in streams:
        total = sum(k for _, k in s["events"])
        inside = sum(k for t, k in s["events"] if w0 < t <= w1)
        win += s["tokens"] * inside / total
    rates = [stream_rate(s) for s in streams]
    frac = overlap / mean_span if mean_span > 0 else 0.0
    return {"ok": overlap >= min_s and frac >= min_frac, "overlap_s": overlap, "overlap_frac": frac,
            "window_tokens": win, "aggregate_tok_s": win / overlap if overlap > 0 else None,
            "per_stream_tok_s": sum(r for r in rates if r) / len([r for r in rates if r]) if any(rates) else None,
            "prefill_wall_s": max(s["events"][0][0] for s in streams) - min(s["t0"] for s in streams)}


def spec_summary(replies: list[dict]) -> dict | None:
    """Drafted / accepted / rounds over replies (bench.reply records); None when no reply carries usage.spec."""
    sp = [r["spec"] for r in replies if r.get("spec")]
    if not sp:
        return None
    d, a, rd = sum(s["drafted"] for s in sp), sum(s["accepted"] for s in sp), sum(s["rounds"] for s in sp)
    n = sum(r["completion_tokens"] for r in replies if r.get("spec"))
    return {"drafted": d, "accepted": a, "rounds": rd, "acceptance": round(a / d, 4) if d else None,
            "tokens_per_round": round(n / rd, 4) if rd else None}


def write_json(path: str, obj) -> None:
    import json
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)


def load_json(path: str):
    import json
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def say(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)
