#!/usr/bin/env python3
"""First-boot smoke: five requests against the server's OpenAI API; raw replies and wall seconds to OUT."""
import json, sys, time, urllib.request
URL, OUT, M = sys.argv[1], sys.argv[2], "inkling-2-mini-1x-dgx-spark"
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
REQ = [
 ("chat-effort-none", "chat/completions", {"messages": [{"role": "user", "content": "What is the capital of France? Answer in one word."}], "max_tokens": 64, "reasoning_effort": "none"}),
 ("chat-effort-high", "chat/completions", {"messages": [{"role": "user", "content": "What is 17 * 23? Think it through, then give the number."}], "max_tokens": 700, "reasoning_effort": "high"}),
 ("completion", "completions", {"prompt": "The three primary colors are", "max_tokens": 24}),
 ("tool-call", "chat/completions", {"messages": [{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}], "tools": TOOLS, "max_tokens": 200, "reasoning_effort": "none"}),
 ("decode-128", "completions", {"prompt": "Once upon a time", "max_tokens": 128}),
]
for name, ep, body in REQ:
    body = {"model": M, "temperature": 0, **body}
    t = time.time()
    r = urllib.request.urlopen(urllib.request.Request(f"{URL}/{ep}", json.dumps(body).encode(), {"content-type": "application/json"}), timeout=900)
    raw = r.read(); dt = time.time() - t
    d = json.loads(raw)
    json.dump({"request": body, "reply": d, "wall_s": round(dt, 3)}, open(f"{OUT}/{name}.json", "w"), indent=1)
    ch = d["choices"][0]; m = ch.get("message", {})
    print(f"== {name}  {dt:.2f}s  finish={ch.get('finish_reason')}  usage={d.get('usage', {}).get('prompt_tokens')}/{d.get('usage', {}).get('completion_tokens')}")
    print("  content:", repr(m.get("content", ch.get("text")))[:400])
    if m.get("reasoning_content"): print("  reasoning:", repr(m["reasoning_content"])[:300])
    if m.get("tool_calls"): print("  tool_calls:", json.dumps(m["tool_calls"])[:300])
