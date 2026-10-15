"""
v8 — Real-use trace log.

Every agent turn appends one JSON line to .agent-cache/traces.jsonl, so the
eval set can grow from what actually goes wrong in use (evals/mine_traces.py
lists candidates) instead of from guesses.

WHAT IS LOGGED, AND WHAT NEVER IS

Logged: timestamp, mode, model, the user's prompt, and per tool call the
tool name, the argument NAMES, and an outcome (ok / error / denied). Plus
the step count, whether MAX_ITERATIONS was hit, and whether the turn used
no tool at all.

Never logged: argument values, tool results, file contents, or the model's
answer. Those are where secrets and personal data live; an eval case needs
the prompt and the shape of what happened, not the data.

The file is local only: .agent-cache/ is git-ignored and on the file tools'
denylist, so the model can neither read nor edit it. --no-trace turns it off.
"""

import datetime
import json
import pathlib

import files

TRACE_FILE = ".agent-cache/traces.jsonl"
ENABLED = True   # agent.py / server.py --no-trace set this False


def outcome(result: str) -> str:
    """Classify a tool result string the way the tools phrase them."""
    head = (result or "").lstrip()
    if head.startswith(("DENIED", "REFUSED")):
        return "denied"
    if head.startswith("Error"):
        return "error"
    return "ok"


class TurnTrace:
    """Collects one turn's shape; write() appends it as a single line."""

    def __init__(self, prompt: str, mode: str, model: str):
        self.record = {
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            "model": model,
            "prompt": prompt,
            "calls": [],
            "steps": 0,
            "max_iterations_hit": False,
            "no_tool_calls": True,
            "ollama_error": False,
        }

    def call(self, tool: str, args, result: str) -> None:
        keys = sorted(args) if isinstance(args, dict) else []
        self.record["calls"].append({"tool": tool, "arg_keys": keys, "outcome": outcome(result)})
        self.record["no_tool_calls"] = False

    def write(self) -> None:
        if not ENABLED:
            return
        path = files.WORKSPACE / TRACE_FILE
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(self.record, ensure_ascii=False) + "\n")
        except OSError:
            pass   # a trace must never break a turn
