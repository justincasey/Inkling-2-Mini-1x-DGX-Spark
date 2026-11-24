import json, urllib.request, sys
U="http://node-4:8000/v1/chat/completions"
P=["What is 17*23? Reply with the integer only.","A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?","What is the capital of France? Answer in one word.","Write a haiku about rain."]
print("effort", *[p[:20] for p in P], sep="\t")
for e in ["none","minimal","low","medium","high","xhigh"]:
    row=[]
    for p in P:
        b={"model":"inkling-2-mini-1x-dgx-spark","messages":[{"role":"user","content":p}],"reasoning_effort":e,"temperature":0,"max_tokens":3000}
        d=json.load(urllib.request.urlopen(urllib.request.Request(U,json.dumps(b).encode(),{"content-type":"application/json"}),timeout=900))
        u=d["usage"]; row.append(f"{u['completion_tokens_details']['reasoning_tokens']}/{u['completion_tokens']}{'L' if d['choices'][0]['finish_reason']=='length' else ''}")
    print(e,*row,sep="\t",flush=True)
