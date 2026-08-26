#!/usr/bin/env python3
"""
bench.py — find out WHY the agent is slow.

Ollama's /api/chat response carries timing fields that most people never
look at. They split a request into two very different phases:

  PROMPT EVAL ("prefill")  — reading everything you sent: system prompt,
                             tool schemas, history, tool results. Processed
                             in parallel, so it is fast per token, but it
                             scales with how much you send.

  EVAL ("generation")      — writing the reply, one token at a time. This is
                             the sequential part and usually dominates.

Knowing which one is slow tells you what to fix:
  slow prompt eval  -> you are sending too much (tool schemas, long results)
  slow generation   -> the model is writing too much (thinking mode!) or the
                       model is simply too big for the hardware

Run:
    python3 bench.py                 # full sweep
    python3 bench.py --model qwen3:8b
    python3 bench.py --quick         # skip the thinking-mode test (slow)
"""

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"

try:
    from tools import TOOL_SCHEMAS
except ImportError:
    TOOL_SCHEMAS = []
    print("[warn] could not import tools.py — tool-schema tests will be skipped")

NS = 1_000_000_000  # Ollama reports nanoseconds


def call(model, messages, *, tools=None, num_ctx=4096, think=False, timeout=600):
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"num_ctx": num_ctx},
        "think": think,
    }
    if tools:
        body["tools"] = tools

    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    wall = time.time() - t0
    data["_wall"] = wall
    return data


def report(label, data):
    """Break one response into its phases."""
    total = data.get("total_duration", 0) / NS
    load = data.get("load_duration", 0) / NS
    p_count = data.get("prompt_eval_count", 0)
    p_dur = data.get("prompt_eval_duration", 0) / NS
    e_count = data.get("eval_count", 0)
    e_dur = data.get("eval_duration", 0) / NS

    p_rate = p_count / p_dur if p_dur else 0
    e_rate = e_count / e_dur if e_dur else 0

    print(f"\n  {label}")
    print(f"    total wall time   : {data['_wall']:.1f}s")
    if load > 0.1:
        print(f"    model load        : {load:.1f}s   (one-off; 0 if already warm)")
    print(f"    prompt eval       : {p_dur:6.1f}s  {p_count:5d} tokens in   ({p_rate:6.1f} tok/s)")
    print(f"    generation        : {e_dur:6.1f}s  {e_count:5d} tokens out  ({e_rate:6.1f} tok/s)")

    if total > 0:
        share = (e_dur / total * 100) if total else 0
        print(f"    generation is {share:.0f}% of total time")
    return {"wall": data["_wall"], "p_count": p_count, "e_count": e_count,
            "p_rate": p_rate, "e_rate": e_rate, "e_dur": e_dur, "p_dur": p_dur}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--quick", action="store_true", help="skip slow thinking test")
    args = ap.parse_args()

    Q = "What is the capital of France? Answer in one sentence."
    msgs = [{"role": "user", "content": Q}]

    print("=" * 68)
    print(f"  BENCHMARK — {args.model}")
    print("=" * 68)

    # ---- warm up so model-load time does not pollute the first result ----
    print("\nwarming up model (loading into memory)...")
    try:
        warm = call(args.model, [{"role": "user", "content": "hi"}])
    except urllib.error.URLError:
        sys.exit("ERROR: cannot reach Ollama at localhost:11434. Is it running?")
    except Exception as exc:
        sys.exit(f"ERROR: {exc}")
    print(f"  loaded in {warm.get('load_duration', 0)/NS:.1f}s")

    results = {}

    # ---- 1. baseline: no tools, small context, no thinking ----
    print("\n" + "-" * 68)
    print("TEST 1 — baseline (no tools, ctx 4096, thinking OFF)")
    print("-" * 68)
    print("  This is your model's raw ceiling. Everything else is overhead.")
    d = call(args.model, msgs, num_ctx=4096, think=False)
    results["baseline"] = report("baseline", d)

    # ---- 2. add tool schemas ----
    if TOOL_SCHEMAS:
        print("\n" + "-" * 68)
        print(f"TEST 2 — with {len(TOOL_SCHEMAS)} tool schemas attached")
        print("-" * 68)
        print("  Same question. Any extra prompt-eval tokens are the schema cost.")
        d = call(args.model, msgs, tools=TOOL_SCHEMAS, num_ctx=4096, think=False)
        results["tools"] = report("with tools", d)
        extra = results["tools"]["p_count"] - results["baseline"]["p_count"]
        print(f"    --> tool schemas cost ~{extra} prompt tokens on EVERY request")

    # ---- 3. bigger context window ----
    print("\n" + "-" * 68)
    print("TEST 3 — ctx 16384 (the value we set as default)")
    print("-" * 68)
    print("  Same input. Slower here means KV-cache allocation is hurting you.")
    d = call(args.model, msgs, tools=TOOL_SCHEMAS or None, num_ctx=16384, think=False)
    results["ctx16k"] = report("ctx 16384", d)

    # ---- 4. thinking mode ----
    if not args.quick:
        print("\n" + "-" * 68)
        print("TEST 4 — thinking mode ON (this is the slow one; be patient)")
        print("-" * 68)
        print("  Watch 'tokens out'. Reasoning tokens are generated at the same")
        print("  speed as visible ones, so they cost real wall-clock time.")
        d = call(args.model, msgs, tools=TOOL_SCHEMAS or None, num_ctx=16384, think=True)
        results["think"] = report("thinking ON", d)

    # ---- 5. thermal check: repeat baseline 3x ----
    print("\n" + "-" * 68)
    print("TEST 5 — sustained load (3 identical calls)")
    print("-" * 68)
    print("  The Air is fanless. If rates DROP across these three, you are")
    print("  thermally throttling and no software change will fix it.")
    rates = []
    for i in range(3):
        d = call(args.model, msgs, num_ctx=4096, think=False)
        e_rate = d.get("eval_count", 0) / (d.get("eval_duration", 1) / NS)
        rates.append(e_rate)
        print(f"    run {i+1}: {e_rate:.1f} tok/s")
    if len(rates) == 3 and rates[0] > 0:
        drop = (rates[0] - rates[-1]) / rates[0] * 100
        verdict = "THROTTLING LIKELY" if drop > 20 else "stable"
        print(f"    --> {drop:+.0f}% change across runs: {verdict}")

    # ---- verdict ----
    print("\n" + "=" * 68)
    print("  VERDICT")
    print("=" * 68)
    base = results["baseline"]
    print(f"\n  Raw generation speed: {base['e_rate']:.1f} tok/s")
    if base["e_rate"] >= 20:
        print("    HEALTHY for an 8B on Apple Silicon.")
    elif base["e_rate"] >= 10:
        print("    SLUGGISH but workable. Model may be spilling to system RAM.")
    else:
        print("    TOO SLOW. Check: is another app using memory? Is the model")
        print("    quantized (q4_K_M)? Run: ollama list  and check the size.")

    if "think" in results:
        t, b = results["think"], results.get("tools", base)
        if b["e_count"]:
            mult = t["e_count"] / b["e_count"]
            print(f"\n  Thinking mode generated {mult:.1f}x more tokens "
                  f"({b['e_count']} -> {t['e_count']})")
            print(f"    and took {t['wall']/b['wall']:.1f}x the wall time.")
            print("    --> This is why --think is off by default.")

    if "ctx16k" in results and "tools" in results:
        c, t = results["ctx16k"], results["tools"]
        delta = (c["wall"] - t["wall"]) / t["wall"] * 100 if t["wall"] else 0
        print(f"\n  Raising ctx 4096 -> 16384 changed wall time by {delta:+.0f}%")
        if delta > 25:
            print("    --> Meaningful cost. Consider --ctx 8192 as a compromise.")
        else:
            print("    --> Cheap. Keep the larger window.")

    print("\n  Remember: in a ReAct loop you pay these costs on EVERY iteration.")
    print("  A 2-tool turn = 3 Ollama calls = 3x everything above.\n")


if __name__ == "__main__":
    main()
