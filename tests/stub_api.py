"""A deterministic stand-in for the server's OpenAI surface, for the static tests (stdlib only, loopback only, no GPU).

`serve()` starts a threaded HTTP server on 127.0.0.1 (an ephemeral port) and yields its URL. It answers /v1/models, /tokenize
(one id per whitespace word), /v1/completions (greedy ids derived from the prompt, so a prompt always gets the same reply;
`usage` with prompt and completion tokens, cached tokens and a `spec` block) and /metrics (the request histograms the bench
tools read). `bug` makes it misbehave on purpose, so a test can show that a gate notices:

  bug="all"    every reply has one id changed (an engine that no longer matches its pinned replies)
  bug="conc"   a reply produced while another request is in flight has one id changed (concurrent != solo)
  bug="short"  every reply ends 5 tokens early with finish_reason "stop" (a short reply)

`State.timeline` holds (start, end) monotonic times of every completion and `State.max_inflight` the most requests ever in
flight at once.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

EOS = 200006


class State:
    def __init__(self, bug: str | None, spec: bool, model: str):
        self.bug, self.spec, self.model = bug, spec, model
        self.lock = threading.Lock()
        self.inflight = 0
        self.max_inflight = 0
        self.timeline: list[tuple[float, float]] = []
        self.ttft_sum = self.e2e_sum = 0.0
        self.count = 0
        self.requests: list[dict] = []


def words_to_ids(text: str) -> list[int]:
    """One id per whitespace word, two for a word with a digit or punctuation (a real tokenizer splits those too)."""
    out = []
    for w in text.split():
        out.append(zlib.crc32(w.encode()) % 190000)
        if any(not c.isalpha() for c in w):
            out.append(zlib.crc32(w[::-1].encode()) % 190000)
    return out


def reply_ids(prompt, n: int) -> list[int]:
    seed = int(hashlib.sha256(json.dumps(prompt).encode()).hexdigest()[:8], 16)
    return [(seed + i * 7919) % 190000 for i in range(n)]


def make_handler(st: State):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence
            pass

        def send(self, status: int, body, ctype="application/json"):
            raw = body.encode() if isinstance(body, str) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/v1/models":
                self.send(200, {"data": [{"id": st.model, "prefix_cache_reach_tokens": 1000000}]})
            elif self.path == "/metrics":
                with st.lock:
                    text = (f"rexl3_request_ttft_seconds_sum {st.ttft_sum}\nrexl3_request_ttft_seconds_count {st.count}\n"
                            f"rexl3_request_e2e_seconds_sum {st.e2e_sum}\nrexl3_request_e2e_seconds_count {st.count}\n")
                self.send(200, text, "text/plain")
            else:
                self.send(404, {"error": "not found"})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/tokenize":
                self.send(200, {"tokens": words_to_ids(body["text"])})
            elif self.path == "/v1/completions":
                with st.lock:
                    st.inflight += 1
                    st.max_inflight = max(st.max_inflight, st.inflight)
                    t_in = time.monotonic()
                    overlapped = st.inflight > 1
                    st.requests.append(body)
                time.sleep(0.03)  # long enough for requests started together to overlap
                with st.lock:
                    overlapped = overlapped or st.inflight > 1
                prompt = body["prompt"]
                n_prompt = len(prompt) if isinstance(prompt, list) else len(words_to_ids(prompt))
                n = int(body.get("max_tokens", 16))
                ids = reply_ids(prompt, n)
                if ids and (st.bug == "all" or (st.bug == "conc" and overlapped)):
                    ids[min(3, len(ids) - 1)] ^= 1
                finish = "length"
                if st.bug == "short":
                    n, finish = max(1, n - 5), "stop"
                    ids = ids[:n]
                spec = {"drafted": 0, "accepted": 0, "rounds": n}
                if st.spec and body.get("draft", True) is not False:
                    spec = {"drafted": n // 2, "accepted": n // 4, "rounds": max(1, n // 3)}
                with st.lock:
                    st.inflight -= 1
                    st.timeline.append((t_in, time.monotonic()))
                    st.count += 1
                    ttft = 0.1 + n_prompt / 1e6
                    st.ttft_sum += ttft
                    st.e2e_sum += ttft + n * 0.01
                self.send(200, {"choices": [{"token_ids": ids, "finish_reason": finish, "text": ""}],
                                "usage": {"prompt_tokens": n_prompt, "completion_tokens": n,
                                          "prompt_tokens_details": {"cached_tokens": 0}, "spec": spec}})
            else:
                self.send(404, {"error": "not found"})

    return H


@contextlib.contextmanager
def serve(bug: str | None = None, spec: bool = True, model: str = "stub-model"):
    """Yield (url, state) of a running stub."""
    st = State(bug, spec, model)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(st))
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}", st
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(timeout=5)
