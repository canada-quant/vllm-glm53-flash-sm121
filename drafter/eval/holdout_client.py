#!/usr/bin/env python3
"""Acceptance-length measurement against a running vLLM spec-decode server (runs inside container dspark2).
Protocol (matches the H200 bench + incoai comparison): holdout prompts rendered with the target chat template
(thinking ON), /v1/completions, temperature 1.0, top_p 0.95, max_tokens 1024, fixed concurrency; acceptance
read from vLLM /metrics deltas (spec_decode_num_drafts / num_draft_tokens / num_accepted_tokens / per_pos).
Writes a JSON result and prints one summary line."""
import argparse, asyncio, json, re, time, urllib.request
import aiohttp
from transformers import AutoTokenizer
ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8100"); ap.add_argument("--prompts", required=True)
ap.add_argument("--limit", type=int, default=500); ap.add_argument("--concurrency", type=int, default=16)
ap.add_argument("--max-tokens", type=int, default=1024); ap.add_argument("--model-dir", default="/data/models/glm53-flash-w4a16-mtp")
ap.add_argument("--served-name", default="glm53"); ap.add_argument("--out", required=True); ap.add_argument("--label", default="")
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model_dir)
rows = [json.loads(l) for l in open(a.prompts)][: a.limit]
def metrics():
    txt = urllib.request.urlopen(a.url + "/metrics", timeout=30).read().decode()
    m = {"drafts": 0.0, "draft_tokens": 0.0, "accepted": 0.0, "per_pos": {}}
    for line in txt.splitlines():
        if line.startswith("#"): continue
        mm = re.match(r"(vllm:spec_decode_num_\w+)(\{[^}]*\})?\s+([0-9.eE+-]+)", line)
        if not mm: continue
        name, labels, val = mm.group(1), mm.group(2) or "", float(mm.group(3))
        if name == "vllm:spec_decode_num_drafts_total": m["drafts"] += val
        elif name == "vllm:spec_decode_num_draft_tokens_total": m["draft_tokens"] += val
        elif name == "vllm:spec_decode_num_accepted_tokens_total": m["accepted"] += val
        elif name == "vllm:spec_decode_num_accepted_tokens_per_pos_total" or name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            pos = re.search(r'position="(\d+)"', labels); 
            if pos: m["per_pos"][int(pos.group(1))] = m["per_pos"].get(int(pos.group(1)), 0.0) + val
    return m
before = metrics(); t0 = time.time(); out_tokens = 0; n_ok = 0; fails = 0
async def one(sess, sem, r):
    global out_tokens, n_ok, fails
    prompt = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    body = {"model": a.served_name, "prompt": prompt, "max_tokens": a.max_tokens, "temperature": 1.0, "top_p": 0.95}
    async with sem:
        try:
            async with sess.post(a.url + "/v1/completions", json=body, timeout=aiohttp.ClientTimeout(total=1200)) as resp:
                j = await resp.json()
            out_tokens += j["usage"]["completion_tokens"]; n_ok += 1
        except Exception as e:
            fails += 1; print("FAIL", r.get("id"), e, flush=True)
async def main():
    sem = asyncio.Semaphore(a.concurrency)
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=a.concurrency + 8)) as sess:
        await asyncio.gather(*(one(sess, sem, r) for r in rows))
asyncio.run(main()); dt = time.time() - t0; after = metrics()
drafts = after["drafts"] - before["drafts"]; dtok = after["draft_tokens"] - before["draft_tokens"]; acc = after["accepted"] - before["accepted"]
per_pos = {p: after["per_pos"].get(p, 0) - before["per_pos"].get(p, 0) for p in sorted(after["per_pos"])}
res = {"label": a.label, "url": a.url, "prompts": len(rows), "ok": n_ok, "fail": fails, "concurrency": a.concurrency, "max_tokens": a.max_tokens,
       "wall_s": round(dt, 1), "output_tokens": out_tokens, "output_tok_per_s": round(out_tokens / dt, 1),
       "drafts": drafts, "draft_tokens": dtok, "accepted_tokens": acc,
       "mean_acceptance_length": round(1 + acc / drafts, 3) if drafts else None,
       "draft_token_acceptance_rate": round(acc / dtok, 4) if dtok else None,
       "per_position_acceptance_rate": {p: round(v / drafts, 4) for p, v in per_pos.items()} if drafts else {}}
json.dump(res, open(a.out, "w"), indent=2)
print(f"RESULT {a.label}: accept_len={res['mean_acceptance_length']} rate={res['draft_token_acceptance_rate']} ok={n_ok}/{len(rows)} out_tok/s={res['output_tok_per_s']} per_pos={res['per_position_acceptance_rate']}", flush=True)
