import json, urllib.request
U="http://node-4:8000"
def call(path, body):
    r=urllib.request.Request(U+path, json.dumps(body).encode(), {"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(r, timeout=300) as f: return f.status, json.loads(f.read())
    except urllib.error.HTTPError as e: return e.code, e.read().decode()[:400]
M="inkling-2-mini-1x-dgx-spark"
msgs=[{"role":"user","content":"Reply with the single word: OK"}]
for extra in ({}, {"reasoning_effort":"none"}):
    s,o=call("/v1/chat/completions/render", {"model":M,"messages":msgs, **extra}); print("render",extra,s,{k:(v if k!='prompt' and k!='text' else v) for k,v in o.items()} if isinstance(o,dict) else o)
    s,o=call("/v1/chat/completions", {"model":M,"messages":msgs,"max_tokens":4, **extra}); print("chat",extra,o["usage"]["prompt_tokens"])
s,o=call("/v1/chat/completions", {"model":M,"messages":msgs,"max_tokens":262145}); print("over", s, str(o)[:600])
s,o=call("/v1/chat/completions", {"model":M,"messages":msgs,"max_tokens":10000000}); print("over2", s, str(o)[:300])
s,o=call("/v1/chat/completions", {"model":M,"messages":[{"role":"user","content":"Write the five words ALPHA BRAVO CHARLIE DELTA ECHO in that order, separated by spaces, and nothing else."}],"max_tokens":512,"reasoning_effort":"none","stop":["CHARLIE"],"temperature":0})
print("stop", s, json.dumps(o["choices"][0], ensure_ascii=False)[:1500], o["usage"])
