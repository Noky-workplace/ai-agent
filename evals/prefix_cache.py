#!/usr/bin/env python3
"""
Does Ollama actually reuse the KV cache for the agent's fixed prefix?

    python3 -m evals.prefix_cache
    python3 -m evals.prefix_cache --model qwen3.5:9b

WHY THIS MATTERS

The decision to keep every tool schema loaded on every turn, rather than
loading tools on demand, rests on one assumption: the system prompt plus
schemas (~1,800+ tokens) is a stable prefix that llama.cpp caches, so after
the first request it costs milliseconds, not seconds. If that's false — an
old Ollama, a chat template that rewrites history between turns — the
premise collapses and dynamic tool loading moves back onto the table.

Ollama v0.30.8 (June 2026) reworked prompt caching and KV reuse. There is no
flag for it, so the only way to know is to measure.

WHAT IT DOES

Three requests, same model, same system prompt and tools:
  1. cold      "Say OK."              — pays the full prefill
  2. repeat    "Say OK."              — identical; best case for the cache
  3. new turn  "Say READY."           — same prefix, different last message:
                                         the case that matters in a real loop
If 2 and 3 prefill in a small fraction of 1's time, the prefix is cached.
"""

import argparse
import pathlib
import sys
import uuid

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.run_evals import _load_tools_and_prompt, make_ollama_chat  # noqa: E402

# A prefill under this fraction of the cold one counts as a cache hit. Loose
# on purpose: the new-turn request must still process its own last message.
HIT_RATIO = 0.35


def verdict(cold_s: float, warm_s: float) -> str:
    if cold_s <= 0:
        return "unknown (no prefill timing reported)"
    ratio = warm_s / cold_s
    word = "CACHED" if ratio < HIT_RATIO else "NOT cached"
    return f"{word}  ({ratio:.0%} of cold prefill)"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Check KV prefix-cache reuse in Ollama.")
    p.add_argument("--model", default="qwen3:8b")
    p.add_argument("--mode", default="general")
    p.add_argument("--ctx", type=int, default=16384)
    p.add_argument("--url", default="http://localhost:11434/api/chat")
    args = p.parse_args(argv)

    prompt, schemas = _load_tools_and_prompt(args.mode, with_memory=False)
    chat = make_ollama_chat(args.model, args.ctx, think=False, temperature=0.0, seed=42,
                            url=args.url)

    def ask(text):
        msgs = [{"role": "system", "content": prompt}, {"role": "user", "content": text}]
        _, t = chat(msgs, schemas)
        return t

    print(f"model {args.model}, {len(schemas)} tools — warming up (load time excluded)...")
    ask("Say OK.")  # absorbs model load so 'cold' measures prefill, not disk
    # A fresh nonce at the very START of the system prompt makes every token
    # after it new to the cache — a genuinely cold prefix. (Changing the end
    # would leave most of the prompt cached and flatter the cold number.)
    cold_prompt = f"[cache probe {uuid.uuid4().hex[:8]}] " + prompt
    msgs = [{"role": "system", "content": cold_prompt}, {"role": "user", "content": "Say OK."}]
    _, cold = chat(msgs, schemas)
    _, repeat = chat(msgs, schemas)
    msgs[-1] = {"role": "user", "content": "Say READY."}
    _, new_turn = chat(msgs, schemas)

    rows = [("cold", cold), ("repeat", repeat), ("new turn", new_turn)]
    print(f"\n{'request':<10} {'prompt tokens':>14} {'prefill s':>10}")
    for name, t in rows:
        print(f"{name:<10} {t['prompt_tokens']:>14} {t['prompt_s']:>10.3f}")
    print()
    print(f"repeat   : {verdict(cold['prompt_s'], repeat['prompt_s'])}")
    print(f"new turn : {verdict(cold['prompt_s'], new_turn['prompt_s'])}")
    print("\n'new turn' is the one that matters: it is what every ReAct step looks like.")
    if new_turn["prompt_s"] >= HIT_RATIO * cold["prompt_s"] > 0:
        print("Not cached: update Ollama (>= 0.30.8) and re-run before trusting the "
              "keep-all-tools-loaded decision.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
