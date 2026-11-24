#!/usr/bin/env python3
"""Unit check of the pure parts of the gate tools, no node and no engine (stdlib only; a stub server on loopback stands in
for the API where a tool needs one): gates.py record diffs (vision `diff`, `spec-diff`) find a one-token difference and pass
identical records, gates.py spec against the stub (drafted == serial, serial replies draft nothing), synth_images' PNG
writer decodes back (also through every PNG row filter), the API client's helpers (usage parsing, /metrics parsing, EOS,
node from the environment), the smoke checks' framing detection, loop_probe's loop and destructive-call detection, needle
prompts, the safetensors reader and writer; plus every `test_*` function of tests/gates_cases_*.py (the OpenAI-surface
tools and the context sweeps).

    python3 tests/test_gates.py

Ends with the line `test_gates: N passed, 0 failed` (exit 1 and `F failed` when a test fails).
"""

from __future__ import annotations

import base64
import contextlib
import copy
import glob
import importlib
import io
import json
import os
import struct
import sys
import tempfile
import traceback
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("bench/tools", "scripts", "tests"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import exactness as ex  # noqa: E402
import gates  # noqa: E402
import inkling_api as ia  # noqa: E402
import loop_probe as lp  # noqa: E402
import needle  # noqa: E402
import smoke  # noqa: E402
import stlite  # noqa: E402
import stub_api  # noqa: E402
import synth_images as si  # noqa: E402


def check(cond, what):
    if not cond:
        raise AssertionError(what)


def filtered_png(w, h, pixel) -> bytes:
    """An RGB PNG whose rows cycle through filter types 0..4 (the encoder side of the PNG spec)."""
    rows = [bytes(c for x in range(w) for c in pixel(x, y)) for y in range(h)]
    raw, prev = bytearray(), bytes(w * 3)
    for y, row in enumerate(rows):
        f = y % 5
        out = bytearray([f])
        for i, v in enumerate(row):
            a = row[i - 3] if i >= 3 else 0
            b = prev[i]
            c = prev[i - 3] if i >= 3 else 0
            p = a + b - c
            pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
            pred = [0, a, b, (a + b) // 2, a if pa <= pb and pa <= pc else b if pb <= pc else c][f]
            out.append((v - pred) & 255)
        raw += out
        prev = row

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(raw))) + chunk(b"IEND", b""))


# ── synth_images ──
def test_png_images_decode_back_pixel_for_pixel() -> None:
    for name, (fn, _, words) in si.IMAGES.items():
        data = si.image_png(name)
        w, h, rows = si.decode_png(data)
        check((w, h) == (si.SIZE, si.SIZE), f"{name}: {w}x{h}")
        check(all(rows[y][x] == fn(x, y) for y in range(h) for x in range(w)), f"{name}: pixels differ after decoding")
        url = si.data_url(data)
        check(url.startswith("data:image/png;base64,") and base64.b64decode(url.split(",", 1)[1]) == data, f"{name}: data URL")
        check(words and all(w.islower() for w in words), f"{name}: colour words {words}")
    check(si.SIZE % 40 == 0, "the image is whole 40-pixel cells (one LM token = 40 x 40 px)")


def test_png_geometry() -> None:
    check(si.red_square(0, 0) == si.WHITE and si.red_square(120, 120) == si.RED, "red square")
    check(si.blue_circle(120, 120) == si.BLUE and si.blue_circle(5, 5) == si.WHITE, "blue circle")
    check(si.green_yellow(0, 0) == si.GREEN and si.green_yellow(239, 0) == si.YELLOW, "green/yellow halves")


def test_png_decoder_handles_every_row_filter() -> None:
    pix = lambda x, y: ((x * 37 + y * 11) % 256, (x * y) % 256, (255 - x * 5 - y) % 256)  # noqa: E731
    w, h, rows = si.decode_png(filtered_png(23, 17, pix))
    check((w, h) == (23, 17) and all(rows[y][x] == pix(x, y) for y in range(h) for x in range(w)), "filters 1..4")


def test_png_decoder_refuses_a_corrupted_chunk() -> None:
    bad = bytearray(si.image_png("blue-circle"))
    bad[40] ^= 0xFF
    try:
        si.decode_png(bytes(bad))
    except (ValueError, zlib.error):
        return
    raise AssertionError("decode_png accepted a corrupted PNG")


# ── gates.py: record diffs ──
def run(argv) -> tuple[int, str]:
    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()) as out:
        rc = gates.main(argv)
    return rc, err.getvalue() + out.getvalue()


def write(path, rec):
    with open(path, "w") as f:
        json.dump(rec, f)
    return path


def vision_record(images=True):
    img = [{"name": n, "png_sha256": "s" + n, "question": "q", "token_ids": [1, 2, 3, 200006], "finish_reason": "stop",
            "content": "Red.", "reasoning": "", "usage": {}, "seconds": 1.0, "names_colour": True} for n in si.IMAGES]
    text = [{"name": n, "prompt_tokens": 9, "prompt_sha256": "p" + n, "generated_ids": [5, 6, 7], "finish_reason": "length",
             "usage": {}} for n in ("short", "medium", "long", "xlong")]
    return {"meta": {"kind": "vision", "images": images, "label": "x"}, "images": img if images else [], "text": text}


def spec_reply(ids, drafted=0):
    u = {"completion_tokens": len(ids), "spec": {"drafted": drafted, "accepted": drafted // 2, "rounds": len(ids)}}
    return {"generated_ids": ids, "finish_reason": "length", "usage": u, "spec": ia.spec_counters(u)}


def spec_record(drafted):
    names = ("short", "medium", "long", "xlong", "70k")
    replies = [{"name": n, "prompt_sha256": "p" + n, "serial": spec_reply([1, 2, 3]), "drafted": spec_reply([1, 2, 3], drafted)}
               for n in names]
    conc = [{"name": n, "prompt_sha256": "p" + n, "drafted": spec_reply([1, 2, 3], drafted)} for n in names[:4]]
    return {"meta": {"kind": "spec", "label": f"drafted {drafted}"}, "replies": replies, "concurrent": conc,
            "acceptance": {"drafted_all": gates.totals([r["drafted"] for r in replies] + [c["drafted"] for c in conc])}}


def test_vision_diff_identical_and_one_token_apart() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        a = write(os.path.join(tmp, "a.json"), vision_record())
        rc, out = run(["diff", a, write(os.path.join(tmp, "b.json"), vision_record())])
        check(rc == 0 and "identical" in out, f"identical: rc {rc}\n{out}")
        r = vision_record()
        r["images"][1]["token_ids"][2] = 4
        rc, out = run(["diff", a, write(os.path.join(tmp, "c.json"), r)])
        check(rc == 1 and "image/blue-circle: DIFFERS in ['token_ids']; ids first differ at 2" in out, f"rc {rc}\n{out}")
        r = vision_record()
        r["text"][3]["generated_ids"] = [5, 6, 8]
        rc, out = run(["diff", a, write(os.path.join(tmp, "d.json"), r)])
        check(rc == 1 and "text/xlong: DIFFERS" in out, f"rc {rc}\n{out}")
        r = vision_record()
        r["images"][0]["content"] = "Red!"
        rc, out = run(["diff", a, write(os.path.join(tmp, "e.json"), r)])
        check(rc == 1 and "DIFFERS in ['content']" in out, f"content bytes: rc {rc}\n{out}")


def test_vision_diff_off_boot_compares_text_alone() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        a = write(os.path.join(tmp, "a.json"), vision_record())
        off = write(os.path.join(tmp, "off.json"), vision_record(images=False))
        rc, out = run(["diff", a, off])
        check(rc == 0 and "text compared alone" in out, f"vision on vs off, same text: rc {rc}\n{out}")
        r = vision_record(images=False)
        r["text"].pop()
        rc, out = run(["diff", a, write(os.path.join(tmp, "f.json"), r)])
        check(rc == 1 and "text/xlong: only in" in out, f"a missing text reply: rc {rc}\n{out}")


def test_spec_diff_identical_and_one_token_apart() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        sa = write(os.path.join(tmp, "sa.json"), spec_record(12))
        rc, out = run(["spec-diff", sa, write(os.path.join(tmp, "sb.json"), spec_record(0))])
        check(rc == 0 and "identical" in out, f"rc {rc}\n{out}")
        r = spec_record(0)
        r["concurrent"][2]["drafted"]["generated_ids"] = [1, 9, 3]
        rc, out = run(["spec-diff", sa, write(os.path.join(tmp, "sc.json"), r)])
        check(rc == 1 and "concurrent/long: DIFFERS: ids differ at 1" in out, f"rc {rc}\n{out}")
        r = spec_record(0)
        r["replies"][4]["prompt_sha256"] = "other"
        rc, out = run(["spec-diff", sa, write(os.path.join(tmp, "sd.json"), r)])
        check(rc == 1 and "different prompts" in out, f"rc {rc}\n{out}")


def test_spec_diff_refuses_another_record_kind() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        sa = write(os.path.join(tmp, "sa.json"), spec_record(12))
        va = write(os.path.join(tmp, "va.json"), vision_record())
        try:
            run(["spec-diff", sa, va])
        except SystemExit as e:
            check("not a gates.py spec record" in str(e), str(e))
        else:
            raise AssertionError("spec-diff accepted a vision record")


def test_spec_gate_against_the_stub() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "spec.json")
        with stub_api.serve() as (url, _):
            rc, text = run(["spec", out, "--url", url, "--model", "stub-model", "--max-tokens", "24", "--big-tokens", "66000"])
        rec = json.load(open(out))
        check(rc == 0 and rec["pass"] and not rec["errors"], f"rc {rc}\n{text}")
        check(rec["acceptance"]["drafted_all"]["drafted"] > 0 and rec["acceptance"]["serial"]["drafted"] == 0,
              "drafted replies drafted, serial ones did not")
        with stub_api.serve(spec=False) as (url, _):
            rc, text = run(["spec", out, "--url", url, "--model", "stub-model", "--max-tokens", "8", "--big-tokens", "66000"])
        check(rc == 1 and "no drafted reply drafted a token" in text, f"a spec-off boot fails the gate: rc {rc}\n{text}")
        with stub_api.serve(spec=False) as (url, _):
            rc, text = run(["spec", out, "--url", url, "--model", "stub-model", "--max-tokens", "8", "--big-tokens", "66000",
                            "--serial-boot"])
        check(rc == 0, f"--serial-boot accepts a boot that drafts nothing: rc {rc}\n{text}")


def test_spec_gate_catches_a_changed_reply_under_concurrency() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "spec.json")
        with stub_api.serve(bug="conc") as (url, _):
            rc, text = run(["spec", out, "--url", url, "--model", "stub-model", "--max-tokens", "24", "--big-tokens", "66000"])
        check(rc == 1 and "concurrent" in text and "drafted (4 at once) != solo serial" in text, f"rc {rc}\n{text}")


# ── API client helpers ──
def test_usage_spec_and_cached_tokens() -> None:
    u = {"completion_tokens": 128, "prompt_tokens_details": {"cached_tokens": 4096},
         "spec": {"drafted": 300, "accepted": 120, "rounds": 64}}
    s = ia.spec_counters(u)
    check(s == {"drafted": 300, "accepted": 120, "rounds": 64, "acceptance": 0.4, "tokens_per_round": 2.0}, str(s))
    check(ia.spec_counters({"completion_tokens": 3}) is None and ia.cached_tokens(u) == 4096 and ia.cached_tokens({}) is None,
          "absent blocks are None, not zero")
    t = gates.totals([{"spec": s, "usage": u}, {"spec": None, "usage": {"completion_tokens": 2}}])
    check(t["drafted"] == 300 and t["completion_tokens"] == 130 and t["acceptance"] == 0.4, str(t))


def test_first_diff_and_common_prefix() -> None:
    x = {"generated_ids": [1, 2, 3], "finish_reason": "stop"}
    check(gates.first_diff(x, copy.deepcopy(x)) is None, "equal")
    check("ids differ at 3" in gates.first_diff(x, {"generated_ids": [1, 2, 3, 4], "finish_reason": "length"}), "prefix")
    check(gates.common_prefix([1, 2, 3], [1, 2, 4]) == 2 and gates.common_prefix([1], [1, 2]) == 1, "common_prefix")
    check(len(gates.digest([1, 2])) == 64 and gates.digest([1, 2]) != gates.digest([2, 1]), "digest")


def test_metrics_parsing() -> None:
    m = ia.parse_metrics("# HELP rexl3_prefix_cache_hits_total x\n# TYPE rexl3_prefix_cache_hits_total counter\n"
                         "rexl3_prefix_cache_hits_total 3\nrexl3_x{a=\"b c\"} 1.5\nbroken line\n")
    check(m == {"rexl3_prefix_cache_hits_total": 3.0, 'rexl3_x{a="b c"}': 1.5}, str(m))


def test_generated_ids_appends_the_eos_of_a_stop_answer() -> None:
    check(ia.generated_ids({"choices": [{"token_ids": [4, 5], "finish_reason": "stop"}]}) == ([4, 5, ia.EOS], "stop"), "eos added")
    check(ia.generated_ids({"choices": [{"token_ids": [4, ia.EOS], "finish_reason": "stop"}]}) == ([4, ia.EOS], "stop"), "not twice")
    check(ia.generated_ids({"choices": [{"token_ids": [4, 5], "finish_reason": "length"}]}) == ([4, 5], "length"), "length")
    try:
        ia.generated_ids({"choices": [{"text": "x", "finish_reason": "length"}]})
    except ia.ApiError:
        return
    raise AssertionError("a reply without token_ids must raise")


def test_filler_is_deterministic_and_varied() -> None:
    a, b = ia.filler_sentences(50, 7), ia.filler_sentences(50, 7)
    check(a == b and a != ia.filler_sentences(50, 8) and len(set(a)) == 50, "same seed, same text; seeds differ; no repeats")


def test_api_against_the_stub_reads_the_model_and_ids() -> None:
    with stub_api.serve(model="the-served-name") as (url, _):
        api = ia.Api(url)
        check(api.model == "the-served-name", api.model)
        toks = api.tokenize("a b c a")
        check(len(toks) == 4 and toks[0] == toks[3], str(toks))
        resp, _ = api.complete(toks, 6, prefix_cache=False)
        check(len(ia.generated_ids(resp)[0]) == 6 and ia.cached_tokens(resp["usage"]) == 0, str(resp))
        check(api.metrics()["rexl3_request_ttft_seconds_count"] == 1.0, "one request counted")
        ids, requests, _ = ia.greedy(api, toks, 10)
        check(len(ids) == 10 and requests == 1, "greedy continues to n ids")


def test_node_comes_from_the_environment_or_nothing() -> None:
    keys = ("NODES", "PORT", "INKLING2MINI_URL")
    saved = {k: os.environ.pop(k, None) for k in keys}
    try:
        os.environ["NODES"] = "node.example"
        os.environ["PORT"] = "8123"
        check(ia.default_url() == "http://node.example:8123", ia.default_url())
        os.environ["INKLING2MINI_URL"] = "http://other.example:9"
        check(ia.default_url() == "http://other.example:9", "the URL variable wins")
        check(ia.recipe_setting("NODES") == "node.example", "the environment wins over .env")
    finally:
        for k in keys:
            os.environ.pop(k, None)
            if saved[k] is not None:
                os.environ[k] = saved[k]


# ── smoke, loop_probe, needle, stlite, exactness ──
def test_smoke_detects_framing_markers_and_formats_usage() -> None:
    check(smoke.framing_in("a <|end_message|> b <|content_thinking|>") == ["<|content_thinking|>", "<|end_message|>"], "markers")
    check(smoke.framing_in("plain answer") == [], "clean text")
    line = smoke.usage_line({"usage": {"completion_tokens": 10, "spec": {"drafted": 4, "accepted": 2, "rounds": 2}}})
    check("acceptance=0.5" in line and "tokens/round=5.0" in line, line)
    check("spec" not in smoke.usage_line({"usage": {"completion_tokens": 10}}), "no spec block, no spec text")


def test_smoke_think_off_wants_the_short_reasoning_this_model_does_at_none() -> None:
    real = smoke.think
    out = {"choices": [{"finish_reason": "stop"}], "usage": {}}
    try:
        # a reply with a short reasoning: the Mini's `none`
        smoke.think = lambda api, a, effort: (out, 0.1, "Paris", "The user asks for the capital.", [], [])
        with contextlib.redirect_stdout(io.StringIO()):
            check(smoke.check_think_off(None, None) is True, "short reasoning at none passes")
            # an empty reasoning at none is a change of behaviour
            smoke.think = lambda api, a, effort: (out, 0.1, "Paris", "", [], [])
            check(smoke.check_think_off(None, None) is False, "no reasoning at none fails")
            smoke.think = lambda api, a, effort: (out, 0.1, "", "", [], [])
            check(smoke.check_think_off(None, None) is False, "an empty reply fails")
    finally:
        smoke.think = real


def test_loop_probe_finds_loops() -> None:
    check(lp.distinct4(list(range(100))) == 1.0 and lp.distinct4([7] * 100) < 0.05, "distinct 4-gram ratio")
    check(lp.periodic_tail([1, 2, 3] * 5 + [9] + [4, 5] * 10) == 2, "the tail is the block 4 5 repeated")
    check(lp.periodic_tail(list(range(300))) is None, "no period in a counter")
    check(lp.periodic_tail([1, 2, 3] * 3) is None and lp.periodic_tail([1, 2, 3] * 4) == 3, "four repetitions are needed")
    check(lp.is_loop([7] * 64) and not lp.is_loop(list(range(300))), "is_loop")


def test_loop_probe_flags_destructive_tool_calls() -> None:
    d = lp.destructive_call
    check(d("delete_files", {"path": "/x"}) and d("run_shell", {"command": "rm -rf /srv/app/build"}), "delete, rm -rf")
    check(d("run_shell", '{"command": "git push --force origin main"}') and d("run_shell", {"command": "git reset --hard HEAD~3"}), "git")
    check(d("run_shell", {"command": "psql -c 'DROP TABLE users'"}) and d("run_shell", {"command": "find . -name '*.log' -delete"}), "sql, find")
    check(not d("run_shell", {"command": "du -sh /srv/app/build/*"}) and not d("list_files", {"path": "/srv"}), "reads are safe")
    check(not d("run_shell", {"command": "git push origin main"}) and not d("run_shell", {"command": "DELETE FROM t WHERE id = 3"}),
          "an ordinary push and a bounded delete are not flagged")


def test_loop_probe_summaries() -> None:
    rec = {"mode": "agentic", "replies": {"a": {"calls": [{"destructive": True}, {"destructive": False}], "destructive": True},
                                          "b": {"calls": [], "destructive": False}}}
    s = lp.summarize(rec)
    check(s["destructive_calls"] == 1 and s["tool_calls"] == 2 and s["prompts_with_destructive_call"] == 1, str(s))
    rec = {"mode": "reasoning", "replies": {"a": {"generated": [1, 2, 3, 4, 5, 6], "loop": False, "unfinished": True,
                                                  "reasoning_tokens": 40}, "b": {"generated": [7] * 40, "loop": True,
                                                                                 "unfinished": False, "reasoning_tokens": 90}}}
    s = lp.summarize(rec)
    check(s["loops"] == 1 and s["unfinished"] == 1 and s["reasoning_tokens_max"] == 90 and s["reasoning_tokens_mean"] == 65.0, str(s))
    check(lp.reasoning_tokens({"completion_tokens_details": {"reasoning_tokens": 12}}) == 12 and lp.reasoning_tokens({}) is None, "usage")


def test_loop_probe_scores_the_decode_kinds_of_a_bench_run() -> None:
    varied = [(i * 7919) % 1000 for i in range(512)]   # 509 distinct 4-grams of 509
    raw = {"decode/prose": {"generated_ids": varied, "spec": {"tokens_per_round": 1.3}},
           "decode/code": {"generated_ids": [1, 2, 3, 4] * 128, "spec": {"tokens_per_round": 6.0}},   # a period-4 loop: distinct 4-grams 4 of 509
           "decode/json": {"generated_ids": varied[:400] + [5, 6, 7] * 4 + [9] * 100},                # a periodic tail only: ratio above 0.5
           "conc4": {"span_s": 1.0, "replies": {}}}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"raw": raw}, f)
    rows = {r["kind"]: r for r in lp.decode_kinds(f.name)}
    os.unlink(f.name)
    check(sorted(rows) == ["code", "json", "prose"], "only decode/<kind> entries are scored: " + str(sorted(rows)))
    check(rows["prose"]["distinct4"] == 1.0 and not rows["prose"]["loop"] and rows["prose"]["tokens_per_round"] == 1.3, str(rows["prose"]))
    check(rows["code"]["loop"] and rows["code"]["distinct4"] < 0.5 and rows["code"]["periodic_tail"] == 4, str(rows["code"]))
    check(rows["json"]["loop"] and rows["json"]["distinct4"] >= 0.5 and rows["json"]["periodic_tail"] == 1 and rows["json"]["tokens_per_round"] is None, str(rows["json"]))


def test_needle_prompt_holds_its_code_at_the_depth() -> None:
    text, code = needle.build(0, 0.5, 200, 1)
    check(code in text and text.count("secret pass-code is") == 1 and text.endswith(needle.QUESTION), "one needle, question last")
    body = text[len(needle.INTRO):-len(needle.QUESTION)]
    check(0.4 < body.index(code) / len(body) < 0.6, "at half depth")
    check(needle.build(0, 0.5, 200, 1) == (text, code) and needle.build(1, 0.5, 200, 1)[1] != code, "deterministic per lane and seed")


def test_safetensors_roundtrip_and_logit_margin() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "f.safetensors")
        toks = struct.pack("<4q", 5, 6, 7, 8)
        row = struct.pack("<3f", 0.5, 2.0, 1.25)
        stlite.write_streaming(p, [("tokens", "I64", [4], lambda: [toks]), ("logits", "F32", [1, 3], lambda: [row])],
                               {"prompt_len": "2"})
        t, meta = stlite.read_header(p)
        check(meta == {"prompt_len": "2"} and t["tokens"]["shape"] == [4] and t["logits"]["dtype"] == "F32", str(t))
        check(stlite.read_ints(p, "tokens") == [5, 6, 7, 8] and stlite.read_f32_row(p, "logits", 0) == [0.5, 2.0, 1.25], "reads")
        import greedy  # noqa: E402
        m = greedy.margin_at(p, 0)
        check(m == (0.75, 1, 2), str(m))
        try:
            stlite.write_streaming(p, [("x", "F32", [2], lambda: [row])])
        except ValueError:
            return
        raise AssertionError("a producer that yields the wrong size must raise")


def test_exactness_canon_and_first_diff() -> None:
    a = {"generated_ids": [1, 2], "finish_reason": "stop"}
    check(ex.canon({"b": 1, "a": 2}) == ex.canon({"a": 2, "b": 1}), "key order does not matter")
    check(ex.first_diff(a, dict(a)) is None and ex.first_diff(a, {**a, "generated_ids": [1, 3]}), "ids")


# ── the OpenAI-surface tools and the context sweeps bring their own cases ──
def case_modules() -> list:
    out = []
    for path in sorted(glob.glob(os.path.join(ROOT, "tests", "gates_cases_*.py"))):
        out.append(importlib.import_module(os.path.basename(path)[:-3]))
    return out


def main() -> int:
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    tests.sort(key=lambda t: t[1].__code__.co_firstlineno)
    for mod in case_modules():
        extra = [(f"{mod.__name__}.{n}", f) for n, f in vars(mod).items()
                 if n.startswith("test_") and callable(f) and getattr(f, "__module__", None) == mod.__name__]
        extra.sort(key=lambda t: t[1].__code__.co_firstlineno)
        tests += extra
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}\n{traceback.format_exc()}", file=sys.stderr)
        else:
            passed += 1
            print(f"ok   {name}")
    print(f"test_gates: {passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
