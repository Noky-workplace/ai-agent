#!/usr/bin/env python3
"""
List real agent turns worth turning into eval cases.

    python3 -m evals.mine_traces                  # reads .agent-cache/traces.jsonl
    python3 -m evals.mine_traces --file other.jsonl

Prints candidates in cases.jsonl format with an EMPTY "expect" for a human
to fill in. It never writes to any eval file: deciding what the right tool
was is the human's job, and a case nobody checked would teach the harness
the agent's own mistakes.

A turn is a candidate when it shows one of:
  - a tool call that returned an error
  - a tool call the user denied
  - MAX_ITERATIONS hit without a final answer
  - no tool at all, for a prompt that sounds time-sensitive
    (today, latest, price, current, now, 今日, 最新, 而家, 價)
"""

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TIME_WORDS = ("today", "latest", "price", "prices", "current", "now")
TIME_WORDS_ZH = ("今日", "最新", "而家", "價")    # no spaces in Chinese: substring match


def triggers(turn: dict) -> list[str]:
    found = []
    outcomes = [c.get("outcome") for c in turn.get("calls", [])]
    if "error" in outcomes:
        found.append("tool error")
    if "denied" in outcomes:
        found.append("denied")
    if turn.get("max_iterations_hit"):
        found.append("max iterations")
    if turn.get("no_tool_calls") and not turn.get("ollama_error"):
        prompt = (turn.get("prompt") or "").lower()
        # Whole words in English, so "know" doesn't match "now".
        words = [w for w in TIME_WORDS if re.search(rf"\b{w}\b", prompt)]
        words += [w for w in TIME_WORDS_ZH if w in prompt]
        if words:
            found.append(f"no tool for a time-sensitive prompt ({', '.join(words)})")
    return found


def mine(lines) -> list[dict]:
    """Candidate cases from trace lines, in order; bad lines are skipped."""
    out, seen = [], set()
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            turn = json.loads(raw)
        except json.JSONDecodeError:
            continue
        why = triggers(turn)
        prompt = turn.get("prompt") or ""
        if not why or not prompt or prompt in seen:
            continue
        seen.add(prompt)
        tools = " -> ".join(f"{c['tool']}({c['outcome']})" for c in turn.get("calls", [])) or "none"
        out.append({
            "id": f"t{len(out) + 1:03d}",
            "category": "trace",
            "prompt": prompt,
            "expect": [],
            "note": f"{'; '.join(why)}. mode {turn.get('mode')}, {turn.get('ts')}. tools: {tools}",
        })
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="List trace turns worth turning into eval cases.")
    p.add_argument("--file", default=None, help="trace file (default .agent-cache/traces.jsonl)")
    args = p.parse_args(argv)
    if args.file:
        path = pathlib.Path(args.file)
    else:
        import files
        import turn_trace
        path = files.WORKSPACE / turn_trace.TRACE_FILE
    if not path.exists():
        print(f"No trace file at {path}. Use the agent without --no-trace first.", file=sys.stderr)
        return 1
    candidates = mine(path.read_text(encoding="utf-8").splitlines())
    for case in candidates:
        print(json.dumps(case, ensure_ascii=False))
    print(f"\n{len(candidates)} candidate(s). Fill in 'expect', give each a real id and "
          f"category, then add the ones you agree with to evals/cases.jsonl by hand.",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
