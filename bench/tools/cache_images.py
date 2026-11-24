#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The prefix cache with images against the node's API (stdlib only, one server): cached == uncached for prompts that carry images.

  cache_images.py --out cache-images.json [--url URL] [--model NAME] [--max-tokens 48]

Chat completions whose user turn is a long filler text and the pack's synthetic images (`synth_images.py`); every request greedy, with
`"prefix_cache": false` for the reference and true for the cached run. Every cached reply must equal its uncached reference, id for id:

  after-text    filler, image, question: uncached; cold with the cache (0 cached tokens); warm (cached tokens, the same reply)
  other-image   the same filler and question under ANOTHER image: its own uncached reply; the cached run equals it, differs from the first
                image's reply, and does not serve the image (cached tokens less than the prompt less its image and question)
  text-first    the filler and question alone are cached first; then the same filler with an image: the text before the image is served
  image-first   image, filler, question: cold and warm; then the same image and filler with another question: the image and filler served
  two-images    filler, image, more text, image, question: cold, warm, and the same prompt with only the second image changed
  concurrent    two image prompts sharing the filler, at the same moment: each equals its uncached reply

Prints `METRIC` lines and the cache's counters; exit 1 on any failure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import inkling_api as ia  # noqa: E402
REFONLY = False
REFS: dict = {}
GOTS: dict = {}
import synth_images as si  # noqa: E402

COUNTERS = ("lookups_total", "hits_total", "tokens_saved_total", "inserts_total", "evictions_total", "restore_failures_total",
            "entries", "snapshot_slots")
QUESTION = "In one short sentence, what colour is the picture, and what shape?"


def counters(api) -> dict:
    m = api.metrics()
    return {k: m.get("rexl3_prefix_cache_" + k) for k in COUNTERS}


def part_text(t: str) -> dict:
    return {"type": "text", "text": t}


def part_image(name: str) -> dict:
    return {"type": "image_url", "image_url": {"url": si.data_url(si.image_png(name))}}


def filler(n: int, seed: int) -> str:
    return " ".join(ia.filler_sentences(n, seed))


def ask(api, parts: list, max_tokens: int, cache: bool) -> dict:
    resp, wall = api.chat([{"role": "user", "content": parts}], max_tokens, reasoning_effort="none", prefix_cache=cache and not REFONLY)
    ch = resp["choices"][0]
    if ch.get("token_ids") is None:
        raise ia.ApiError(200, resp, "/v1/chat/completions (no token_ids)")
    u = resp.get("usage") or {}
    return {"ids": [int(t) for t in ch["token_ids"]], "finish": ch.get("finish_reason"), "cached": ia.cached_tokens(u),
            "prompt_tokens": u.get("prompt_tokens"), "content": (ch.get("message") or {}).get("content") or "", "wall_s": round(wall, 3)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ia.add_api_args(ap)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--reference-only", action="store_true",
                    help="send every request uncached (a server without a cache): the replies a cached run must equal, byte for byte")
    a = ap.parse_args(argv)
    global REFONLY
    REFONLY = a.reference_only
    api = ia.api_from_args(a)
    mt = a.max_tokens
    errs: list[str] = []
    log: dict = {}
    REFS.clear()
    GOTS.clear()
    c0 = counters(api)
    if not c0.get("snapshot_slots") and not REFONLY:
        raise SystemExit("the server runs no prefix cache: boot with PREFIX_CACHE=on")
    names = list(si.IMAGES)
    X, Y = names[0], names[1]
    q = part_text(QUESTION)

    def check(label: str, ref: dict, got: dict, cached_min: int | None = None, cached_max: int | None = None, zero: bool = False) -> None:
        REFS[label], GOTS[label] = ref["ids"], got["ids"]
        if got["ids"] != ref["ids"]:
            errs.append(f"{label}: the cached reply differs from the uncached one ({got['content'][:60]!r} vs {ref['content'][:60]!r})")
        if REFONLY:
            return
        if ref["cached"]:
            errs.append(f"{label}: the uncached reference was served {ref['cached']} cached tokens")
        if zero and got["cached"]:
            errs.append(f"{label}: a cold reply was served {got['cached']} cached tokens")
        if cached_min is not None and (got["cached"] or 0) < cached_min:
            errs.append(f"{label}: {got['cached']} cached tokens (at least {cached_min} expected)")
        if cached_max is not None and (got["cached"] or 0) > cached_max:
            errs.append(f"{label}: {got['cached']} cached tokens (at most {cached_max} expected)")

    # after-text
    F1 = part_text(filler(130, 8101))
    p_x = [F1, part_image(X), q]
    ref = ask(api, p_x, mt, False)
    cold = ask(api, p_x, mt, True)
    warm = ask(api, p_x, mt, True)
    check("after-text cold", ref, cold, zero=True)
    check("after-text warm", ref, warm, cached_min=1000)
    log["after-text"] = {"ref": ref, "cold": cold, "warm": warm}
    print(f"after-text: {ref['prompt_tokens']} prompt tokens; cold cached {cold['cached']}, warm cached {warm['cached']}; equal {cold['ids'] == ref['ids'] == warm['ids']}", flush=True)

    # other-image
    p_y = [F1, part_image(Y), q]
    ref_y = ask(api, p_y, mt, False)
    got_y = ask(api, p_y, mt, True)
    # the cache's end cut is 64 tokens before the end, and the image and the question are inside those 64: what is served is text before the image
    check("other-image", ref_y, got_y, cached_max=(ref_y["prompt_tokens"] or 0) - 64)
    if got_y["ids"] == ref["ids"]:
        errs.append("other-image: another image under the same text gave the first image's reply (the image is not read)")
    log["other-image"] = {"ref": ref_y, "cached": got_y}
    print(f"other-image: cached {got_y['cached']} of {ref_y['prompt_tokens']}; own reply equal {got_y['ids'] == ref_y['ids']}; differs from image {X}'s: {got_y['ids'] != ref['ids']}", flush=True)

    # text-first
    F2 = part_text(filler(130, 8102))
    t_only = [F2, q]
    r_t = ask(api, t_only, mt, False)
    c_t = ask(api, t_only, mt, True)
    c_t2 = ask(api, t_only, mt, True)
    check("text-first (text only, cold)", r_t, c_t, zero=True)
    check("text-first (text only, warm)", r_t, c_t2, cached_min=1000)
    p_i = [F2, part_image(X), q]
    r_i = ask(api, p_i, mt, False)
    c_i = ask(api, p_i, mt, True)
    check("text-first (image after the cached text)", r_i, c_i, cached_min=1000)
    log["text-first"] = {"text_ref": r_t, "text_cold": c_t, "text_warm": c_t2, "image_ref": r_i, "image_cached": c_i}
    print(f"text-first: the image prompt after a cached text-only prefix: cached {c_i['cached']} of {r_i['prompt_tokens']}; equal {c_i['ids'] == r_i['ids']}", flush=True)

    # image-first
    F3 = part_text(filler(130, 8103))
    p_f = [part_image(X), F3, q]
    r_f = ask(api, p_f, mt, False)
    c_f = ask(api, p_f, mt, True)
    w_f = ask(api, p_f, mt, True)
    check("image-first cold", r_f, c_f, zero=True)
    check("image-first warm", r_f, w_f, cached_min=1000)
    q2 = part_text("Describe the picture in a few words.")
    p_f2 = [part_image(X), F3, q2]
    r_f2 = ask(api, p_f2, mt, False)
    c_f2 = ask(api, p_f2, mt, True)
    check("image-first, another question", r_f2, c_f2, cached_min=1000)
    log["image-first"] = {"ref": r_f, "cold": c_f, "warm": w_f, "q2_ref": r_f2, "q2_cached": c_f2}
    print(f"image-first: cold {c_f['cached']} warm {w_f['cached']} other question {c_f2['cached']} cached; equal {c_f['ids'] == r_f['ids'] == w_f['ids'] and c_f2['ids'] == r_f2['ids']}", flush=True)

    # two-images
    mid = part_text(filler(60, 8104))
    p_2 = [F1, part_image(X), mid, part_image(Y), q]
    r_2 = ask(api, p_2, mt, False)
    c_2 = ask(api, p_2, mt, True)
    w_2 = ask(api, p_2, mt, True)
    check("two-images cold", r_2, c_2, cached_min=0)
    check("two-images warm", r_2, w_2, cached_min=1000)
    p_2b = [F1, part_image(X), mid, part_image(names[2]), q]
    r_2b = ask(api, p_2b, mt, False)
    c_2b = ask(api, p_2b, mt, True)
    check("two-images, only the second image changed", r_2b, c_2b, cached_min=1000)
    log["two-images"] = {"ref": r_2, "cold": c_2, "warm": w_2, "b_ref": r_2b, "b_cached": c_2b}
    print(f"two-images: cold {c_2['cached']} warm {w_2['cached']}; second image changed: cached {c_2b['cached']}; equal {w_2['ids'] == r_2['ids'] and c_2b['ids'] == r_2b['ids']}", flush=True)

    # concurrent
    jobs = [[F1, part_image(X), part_text("What colour is it?")], [F1, part_image(Y), part_text("What shape is it?")]]
    refs = [ask(api, j, mt, False) for j in jobs]
    out: list = [None, None]
    barrier = threading.Barrier(2)

    def one(i):
        barrier.wait()
        out[i] = ask(api, jobs[i], mt, True)
    th = [threading.Thread(target=one, args=(i,)) for i in range(2)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    for i in range(2):
        check(f"concurrent {i}", refs[i], out[i], cached_min=1000)
    log["concurrent"] = {"refs": refs, "cached": out}
    print(f"concurrent: two image prompts at once, cached {[o['cached'] for o in out]}; equal {[o['ids'] == r['ids'] for o, r in zip(out, refs)]}", flush=True)

    c1 = counters(api)
    if (c1["restore_failures_total"] or 0) - (c0["restore_failures_total"] or 0):
        errs.append("a restore failed")
    d = {k: (c1[k] or 0) - (c0[k] or 0) for k in ("lookups_total", "hits_total", "tokens_saved_total", "inserts_total", "evictions_total")}
    metrics = {f"cache_images_{k}": v for k, v in d.items()}
    with open(a.out, "w") as f:
        json.dump({"meta": {"model": api.model, "max_tokens": mt, "images": names, "counters_before": c0, "counters_after": c1,
                            "reference_only": REFONLY}, "log": log, "ids": {"refs": REFS, "got": GOTS},
                   "metrics": metrics, "errors": errs}, f, indent=1)
    for k, v in metrics.items():
        print(f"METRIC {k}={v}")
    for e in errs:
        print(f"FAIL {e}")
    print(f"cache_images: {'pass' if not errs else 'FAIL'} -> {a.out}")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
