#!/usr/bin/env python3
"""
v7 — Tool-selection evals.

    python3 -m evals.run_evals                       # baseline, qwen3:8b
    python3 -m evals.run_evals --model qwen3.5:9b    # candidate model
    python3 -m evals.run_evals --mode finance        # a mode's tool subset
    python3 -m evals.run_evals --compare A.json B.json

WHAT IT MEASURES

One thing, measured well: given a user message, does the model's FIRST action
pick the right tool — or correctly pick no tool at all? Plus, when it does
call a tool, are the arguments valid against the schema?

That is the decision this project has already watched fail. An 8B model once
answered from stale training data instead of calling web_search, and the fix
(a bigger context window) was verified by eye. Every "this made the agent
smarter" claim since then has been anecdotal. This turns it into a number,
so a model swap, a prompt edit, or a tool-description rewrite can be judged
by a before/after diff rather than by a good-looking demo.

WHAT IT DELIBERATELY DOES NOT DO

- It never executes a tool. Only the model's first reply is inspected, so a
  run cannot write files, run code, or hit the network beyond Ollama. That
  makes it safe to leave running unattended.
- It is single-step. BFCL's authors note that single-call accuracy overstates
  real agent skill — multi-turn accuracy is much lower for most open models.
  Treat a high score here as necessary, not sufficient.

DETERMINISM

The default is temperature 0 with a fixed seed, so two runs on the same model
and prompt should agree and a diff between runs reflects the change you made,
not sampling noise. The agent itself runs at the model's default temperature;
pass --temperature model --runs 3 to measure that, at 3x the cost.
"""

import argparse
import collections
import datetime
import hashlib
import json
import pathlib
import statistics
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OLLAMA_URL = "http://localhost:11434/api/chat"
CASES_FILE = pathlib.Path(__file__).with_name("cases.jsonl")
RESULTS_DIR = pathlib.Path(__file__).with_name("results")
NO_TOOL = "none"
ERROR = "error"
REQUIRED_FIELDS = ("id", "category", "prompt", "expect")

_JSON_TYPES = {
    "string": str,
    "boolean": bool,
    "integer": int,
    "number": (int, float),
    "object": dict,
    "array": list,
}


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------

def load_cases(path: pathlib.Path = CASES_FILE) -> list[dict]:
    """Parse cases.jsonl. Fails loudly on malformed lines or duplicate ids.

    A silently skipped case would shrink the denominator and inflate the
    score, so every problem is an error with a line number.
    """
    cases, seen = [], set()
    for lineno, line in enumerate(pathlib.Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: bad JSON ({exc})") from None
        missing = [f for f in REQUIRED_FIELDS if f not in case]
        if missing:
            raise ValueError(f"{path}:{lineno}: missing {', '.join(missing)}")
        if not isinstance(case["expect"], list) or not case["expect"]:
            raise ValueError(f"{path}:{lineno}: 'expect' must be a non-empty list")
        if case["id"] in seen:
            raise ValueError(f"{path}:{lineno}: duplicate id {case['id']!r}")
        seen.add(case["id"])
        cases.append(case)
    return cases


def file_sha(path: pathlib.Path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:12]


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------
# Inspecting a reply
# --------------------------------------------------------------------------

def schema_index(schemas: list[dict]) -> dict[str, dict]:
    return {s["function"]["name"]: s["function"].get("parameters", {}) for s in schemas}


def first_action(message: dict) -> tuple[str, dict]:
    """The model's first decision: a tool name and its arguments, or NO_TOOL."""
    calls = message.get("tool_calls") or []
    if not calls:
        return NO_TOOL, {}
    fn = calls[0].get("function", {}) or {}
    args = fn.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"__unparseable__": args}
    if not isinstance(args, dict):
        args = {"__not_an_object__": args}
    return fn.get("name", "") or "", args


def check_args(name: str, args: dict, index: dict[str, dict]) -> list[str]:
    """Validate arguments against the tool's JSON schema. Returns problems.

    Not a full JSON-schema validator — it checks the failure modes small
    models actually produce: missing required fields, invented fields,
    wrong primitive types, and values outside an enum.
    """
    if name in (NO_TOOL, ERROR):
        return []
    params = index.get(name)
    if params is None:
        return [f"unknown tool {name!r}"]
    problems = []
    for bad in ("__unparseable__", "__not_an_object__"):
        if bad in args:
            return [f"arguments were not a JSON object: {args[bad]!r}"[:120]]
    props = params.get("properties", {}) or {}
    for req in params.get("required", []) or []:
        if req not in args:
            problems.append(f"missing required {req!r}")
    for key, value in args.items():
        spec = props.get(key)
        if spec is None:
            problems.append(f"unknown argument {key!r}")
            continue
        expected = _JSON_TYPES.get(spec.get("type"))
        # bool is a subclass of int in Python; don't let True pass as a number.
        if expected and (not isinstance(value, expected)
                         or (isinstance(value, bool) and spec.get("type") in ("integer", "number"))):
            problems.append(f"{key!r} should be {spec.get('type')}, got {type(value).__name__}")
        elif "enum" in spec and value not in spec["enum"]:
            problems.append(f"{key!r}={value!r} not in {spec['enum']}")
    return problems


# --------------------------------------------------------------------------
# Talking to Ollama
# --------------------------------------------------------------------------

def make_ollama_chat(model: str, num_ctx: int, think: bool, temperature,
                     seed: int, timeout: float = 180.0, url: str = OLLAMA_URL):
    """Return chat(messages, schemas) -> (message, timings).

    A closure rather than a class so tests can swap in any function with the
    same signature. If the model rejects the 'think' field (models without a
    reasoning mode can), retry once without it and remember that.
    """
    state = {"send_think": True}

    def chat(messages, schemas):
        options = {"num_ctx": num_ctx, "seed": seed}
        if temperature is not None:
            options["temperature"] = temperature
        body = {"model": model, "messages": messages, "tools": schemas,
                "stream": False, "options": options}
        if state["send_think"]:
            body["think"] = think
        req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code == 400 and "think" in detail.lower() and state["send_think"]:
                state["send_think"] = False
                return chat(messages, schemas)
            raise RuntimeError(f"HTTP {exc.code}: {detail[:200]}") from None
        ns = 1e-9
        timings = {
            "load_s": data.get("load_duration", 0) * ns,
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "prompt_s": data.get("prompt_eval_duration", 0) * ns,
            "gen_tokens": data.get("eval_count", 0),
            "gen_s": data.get("eval_duration", 0) * ns,
        }
        return data.get("message", {}) or {}, timings

    return chat


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------

def _primary(case: dict) -> str:
    return case["expect"][0]


def run(cases: list[dict], chat, schemas: list[dict], system_prompt: str,
        runs: int = 1, progress=print) -> dict:
    """Score every case. Returns the per-case records and a summary.

    A case is SKIPPED when its primary expected tool isn't in this schema set
    (an older checkout, or a mode that hides it). Skipped cases are excluded
    from the denominator and listed, so a smaller tool set can't look better
    just by dodging questions it can't answer.
    """
    index = schema_index(schemas)
    available = set(index) | {NO_TOOL}
    records, skipped = [], []

    for case in cases:
        if _primary(case) not in available:
            skipped.append(case["id"])
            continue
        accepted = [e for e in case["expect"] if e in available]
        answers = []
        for _ in range(runs):
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": case["prompt"]}]
            t0 = time.perf_counter()
            try:
                message, timings = chat(messages, schemas)
                got, args = first_action(message)
                err = None
            except Exception as exc:  # one bad request must not sink the run
                got, args, timings, err = ERROR, {}, {}, str(exc)[:200]
            wall = time.perf_counter() - t0
            problems = check_args(got, args, index)
            answers.append({"got": got, "args": args, "arg_problems": problems,
                            "wall_s": round(wall, 3), "timings": timings, "error": err})

        hits = [a["got"] in accepted for a in answers]
        record = {
            "id": case["id"],
            "category": case["category"],
            "prompt": case["prompt"],
            "expect": accepted,
            "answers": answers,
            "hit_rate": sum(hits) / len(hits),
            "hit": all(hits),
            "consistent": len({a["got"] for a in answers}) == 1,
        }
        records.append(record)
        if progress:
            first = answers[0]
            tag = "ok  " if record["hit"] else "MISS"
            extra = "" if record["hit"] else f"  expected {'/'.join(accepted)}"
            bad_args = f"  args: {first['arg_problems'][0]}" if first["arg_problems"] else ""
            progress(f"  [{tag}] {case['id']:<5} {first['got']:<16}{extra}{bad_args}")

    return {"cases": records, "skipped": skipped, "summary": summarise(records, runs)}


def summarise(records: list[dict], runs: int = 1) -> dict:
    n = len(records)
    by_cat = collections.defaultdict(list)
    by_tool = collections.defaultdict(list)
    confusion = collections.Counter()
    tool_answers, valid_args = 0, 0
    walls, prompt_s, gen_tok_s = [], [], []

    for r in records:
        by_cat[r["category"]].append(r["hit_rate"])
        by_tool[r["expect"][0]].append(r["hit_rate"])
        for a in r["answers"]:
            if a["got"] not in r["expect"]:
                confusion[f"{r['expect'][0]} -> {a['got']}"] += 1
            if a["got"] not in (NO_TOOL, ERROR):
                tool_answers += 1
                valid_args += not a["arg_problems"]
            walls.append(a["wall_s"])
            t = a.get("timings") or {}
            if t.get("prompt_s"):
                prompt_s.append(t["prompt_s"])
            if t.get("gen_s") and t.get("gen_tokens"):
                gen_tok_s.append(t["gen_tokens"] / t["gen_s"])

    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    return {
        "n": n,
        "runs": runs,
        "accuracy": mean([r["hit_rate"] for r in records]),
        "strict_accuracy": mean([1.0 if r["hit"] else 0.0 for r in records]),
        "consistency": mean([1.0 if r["consistent"] else 0.0 for r in records]) if runs > 1 else None,
        "arg_validity": round(valid_args / tool_answers, 4) if tool_answers else None,
        "errors": sum(a["got"] == ERROR for r in records for a in r["answers"]),
        "by_category": {k: mean(v) for k, v in sorted(by_cat.items())},
        "by_expected_tool": {k: mean(v) for k, v in sorted(by_tool.items())},
        "top_confusions": confusion.most_common(10),
        "latency_s": {
            "mean": mean(walls),
            "median": round(statistics.median(walls), 3) if walls else 0.0,
            "max": round(max(walls), 3) if walls else 0.0,
        },
        "prompt_eval_s_median": round(statistics.median(prompt_s), 3) if prompt_s else None,
        "gen_tok_per_s_median": round(statistics.median(gen_tok_s), 1) if gen_tok_s else None,
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def format_summary(report: dict) -> str:
    s, meta = report["summary"], report.get("meta", {})
    if not s["n"]:
        return (f"\nNo cases scored: {len(report.get('skipped', []))} skipped because their "
                f"primary tool isn't loaded in mode {meta.get('mode', 'general')!r}. "
                f"Try --mode finance for market cases.")
    lines = [
        "",
        f"model {meta.get('model', '?')}   mode {meta.get('mode', 'general')}   "
        f"tools {len(meta.get('tools', []))}   cases {s['n']}"
        f" (+{len(report.get('skipped', []))} skipped)   runs {s.get('runs', 1)}",
        f"accuracy      {s['accuracy']:.1%}",
    ]
    if s.get("consistency") is not None:
        lines.append(f"consistency   {s['consistency']:.1%}   (same answer every run)")
    if s.get("arg_validity") is not None:
        lines.append(f"arg validity  {s['arg_validity']:.1%}   (of tool calls made)")
    if s.get("errors"):
        lines.append(f"errors        {s['errors']}   (requests that failed outright)")
    lat = s["latency_s"]
    lines.append(f"latency       median {lat['median']}s   mean {lat['mean']}s   max {lat['max']}s")
    if s.get("prompt_eval_s_median") is not None:
        lines.append(f"prefill       median {s['prompt_eval_s_median']}s per request")
    if s.get("gen_tok_per_s_median") is not None:
        lines.append(f"decode        median {s['gen_tok_per_s_median']} tok/s")
    lines.append("")
    lines.append("by category")
    for cat, acc in s["by_category"].items():
        lines.append(f"  {cat:<12} {acc:6.1%}")
    if s["top_confusions"]:
        lines.append("")
        lines.append("most common misses (expected -> got)")
        for pair, count in s["top_confusions"]:
            lines.append(f"  {count:>3}  {pair}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Multi-step chains (--chains)
# --------------------------------------------------------------------------
#
# A chain plays 2-3 steps. At each step the model's first action is scored
# exactly as above; if it is right, the step's SCRIPTED result goes back as a
# role:tool message and the next step is scored. No tool ever runs: the
# result is fixed text from chains.jsonl, whatever the model asked for.
#
# A chain may start with "history": scripted earlier calls and their results,
# for testing the reaction to a specific tool result (an error, a refusal).
#
# A miss ends the chain. Its later steps are "not reached" and count as
# misses in per-step accuracy, because a model that never gets to step 2
# hasn't shown it can do step 2.

CHAINS_FILE = pathlib.Path(__file__).with_name("chains.jsonl")


def load_chains(path: pathlib.Path = CHAINS_FILE) -> list[dict]:
    """Parse chains.jsonl, failing loudly like load_cases."""
    chains, seen = [], set()
    for lineno, line in enumerate(pathlib.Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            chain = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: bad JSON ({exc})") from None
        for field in ("id", "category", "prompt", "steps"):
            if field not in chain:
                raise ValueError(f"{path}:{lineno}: missing {field}")
        steps = chain["steps"]
        if not isinstance(steps, list) or len(steps) < 2:
            raise ValueError(f"{path}:{lineno}: a chain needs at least 2 steps")
        for i, step in enumerate(steps):
            if not isinstance(step.get("expect"), list) or not step["expect"]:
                raise ValueError(f"{path}:{lineno}: step {i + 1} needs a non-empty 'expect'")
            last = i == len(steps) - 1
            if not last and "result" not in step:
                raise ValueError(f"{path}:{lineno}: step {i + 1} needs a scripted 'result'")
        for h in chain.get("history", []):
            if not isinstance(h.get("call"), dict) or "name" not in h["call"] or "result" not in h:
                raise ValueError(f"{path}:{lineno}: history entries need call.name and result")
        if chain["id"] in seen:
            raise ValueError(f"{path}:{lineno}: duplicate id {chain['id']!r}")
        seen.add(chain["id"])
        chains.append(chain)
    return chains


def check_step(step: dict, got: str, args: dict, prev: dict | None) -> list[str]:
    """The step's own checks on the arguments, beyond the schema.

      equals   {arg: value}  the argument must equal value
      contains {arg: text}   the argument must contain text
      within   {arg: text}   the argument must be non-empty and appear in text
                             (edit_file's old_string copied from the file)
      forbid   {arg: value}  the argument must NOT equal value
      forbid_repeat          fail if this call repeats the previous step's
                             tool with identical arguments (a refused call
                             retried unchanged)

    "check_by_tool": {tool: {...}} adds checks that apply only when the model
    called that tool, for steps that accept more than one.
    """
    checks = dict(step.get("check", {}) or {})
    for key, value in ((step.get("check_by_tool") or {}).get(got) or {}).items():
        merged = dict(checks.get(key) or {}) if isinstance(value, dict) else None
        checks[key] = {**merged, **value} if merged is not None else value
    problems = []
    for arg, value in (checks.get("equals") or {}).items():
        if args.get(arg) != value:
            problems.append(f"{arg!r} should be {value!r}, got {args.get(arg)!r}")
    for arg, text in (checks.get("contains") or {}).items():
        v = args.get(arg)
        if not isinstance(v, str) or text not in v:
            problems.append(f"{arg!r} should contain {text!r}")
    for arg, text in (checks.get("within") or {}).items():
        v = args.get(arg)
        if not isinstance(v, str) or not v.strip() or v not in text:
            problems.append(f"{arg!r} is not copied from the given text")
    for arg, value in (checks.get("forbid") or {}).items():
        if args.get(arg) == value:
            problems.append(f"{arg!r} must not be {value!r}")
    if checks.get("forbid_repeat") and prev and got == prev["got"] and args == prev["args"]:
        problems.append("repeated the previous call unchanged")
    return problems


def run_chains(chains: list[dict], chat, schemas: list[dict], system_prompt: str,
               progress=print) -> dict:
    """Play every chain against the model with scripted tool results."""
    index = schema_index(schemas)
    available = set(index) | {NO_TOOL}
    records, skipped = [], []

    for chain in chains:
        if any(step["expect"][0] not in available for step in chain["steps"]):
            skipped.append(chain["id"])
            continue
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": chain["prompt"]}]
        steps, prev = [], None
        # Optional scripted earlier calls, e.g. an edit that already failed,
        # so the first scored step is the model's reaction to that result.
        for h in chain.get("history", []):
            args = h["call"].get("arguments", {})
            messages.append({"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": h["call"]["name"], "arguments": args}}]})
            messages.append({"role": "tool", "name": h["call"]["name"], "content": h["result"]})
            prev = {"got": h["call"]["name"], "args": args}
        for i, step in enumerate(chain["steps"]):
            accepted = [e for e in step["expect"] if e in available]
            t0 = time.perf_counter()
            try:
                message, timings = chat(messages, schemas)
                got, args = first_action(message)
                err = None
            except Exception as exc:
                message, got, args, timings, err = {}, ERROR, {}, {}, str(exc)[:200]
            wall = time.perf_counter() - t0
            problems = check_step(step, got, args, prev)
            hit = got in accepted and not problems and err is None
            steps.append({"step": i + 1, "expect": accepted, "got": got, "args": args,
                          "arg_problems": check_args(got, args, index),
                          "check_problems": problems, "hit": hit,
                          "wall_s": round(wall, 3), "timings": timings, "error": err})
            if not hit or got == NO_TOOL or i == len(chain["steps"]) - 1:
                break
            # Feed back only the first call, so one scripted result answers it.
            call = (message.get("tool_calls") or [])[0]
            messages.append({"role": "assistant", "content": message.get("content") or "",
                             "tool_calls": [call]})
            messages.append({"role": "tool", "name": got, "content": step["result"]})
            prev = {"got": got, "args": args}

        n_steps = len(chain["steps"])
        hits = sum(s["hit"] for s in steps)
        records.append({"id": chain["id"], "category": chain["category"],
                        "prompt": chain["prompt"], "n_steps": n_steps,
                        "reached": len(steps), "steps": steps,
                        "hit": hits == n_steps})
        if progress:
            marks = "".join("+" if s["hit"] else "x" for s in steps)
            marks += "." * (n_steps - len(steps))
            last = steps[-1]
            why = "" if hits == n_steps else (
                f"  step {last['step']}: expected {'/'.join(last['expect'])}, got {last['got']}"
                + (f" ({last['check_problems'][0]})" if last["check_problems"] else ""))
            progress(f"  [{marks:<3}] {chain['id']:<5}{why}")

    return {"chains": records, "skipped": skipped, "summary": summarise_chains(records)}


def summarise_chains(records: list[dict]) -> dict:
    """Whole-chain accuracy, per-step accuracy, and the step where chains fail."""
    n = len(records)
    max_steps = max((r["n_steps"] for r in records), default=0)
    per_step, given_reached, failed_at = {}, {}, collections.Counter()
    for k in range(1, max_steps + 1):
        has = [r for r in records if r["n_steps"] >= k]
        hit = [r for r in has if len(r["steps"]) >= k and r["steps"][k - 1]["hit"]]
        reached = [r for r in has if len(r["steps"]) >= k]
        per_step[str(k)] = round(len(hit) / len(has), 4) if has else None
        given_reached[str(k)] = round(len(hit) / len(reached), 4) if reached else None
    by_cat = collections.defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r["hit"])
        if not r["hit"]:
            last = r["steps"][-1]
            failed_at[f"step {last['step']}: {'/'.join(last['expect'])} -> {last['got']}"] += 1
    all_steps = [s for r in records for s in r["steps"]]
    return {
        "n": n,
        "chain_accuracy": round(sum(r["hit"] for r in records) / n, 4) if n else 0.0,
        "step_accuracy": per_step,
        "step_accuracy_given_reached": given_reached,
        "by_category": {c: round(sum(v) / len(v), 4) for c, v in sorted(by_cat.items())},
        "failures": failed_at.most_common(),
        "errors": sum(1 for s in all_steps if s["error"]),
        "latency_s": {"median": round(statistics.median([s["wall_s"] for s in all_steps]), 3)
                      if all_steps else 0.0},
    }


def format_chain_summary(report: dict) -> str:
    s, meta = report["summary"], report.get("meta", {})
    if not s["n"]:
        return f"\nNo chains scored: {len(report.get('skipped', []))} skipped (tools not in this mode)."
    lines = [
        "",
        f"model {meta.get('model', '?')}   mode {meta.get('mode', 'general')}   "
        f"chains {s['n']} (+{len(report.get('skipped', []))} skipped)",
        f"whole-chain accuracy  {s['chain_accuracy']:.1%}",
        "per step              " + "   ".join(
            f"{k}: {v:.1%}" for k, v in s["step_accuracy"].items() if v is not None),
        "per step, if reached  " + "   ".join(
            f"{k}: {v:.1%}" for k, v in s["step_accuracy_given_reached"].items() if v is not None),
        f"median step latency   {s['latency_s']['median']}s",
    ]
    if s.get("errors"):
        lines.append(f"errors                {s['errors']}")
    lines.append("")
    lines.append("by category")
    for cat, acc in s["by_category"].items():
        lines.append(f"  {cat:<14} {acc:6.1%}")
    if s["failures"]:
        lines.append("")
        lines.append("where chains failed")
        for what, count in s["failures"]:
            lines.append(f"  {count:>3}  {what}")
    return "\n".join(lines)


def compare(a: dict, b: dict) -> str:
    """Diff two reports: what got better, what broke.

    Only cases present (not skipped) in BOTH runs are compared, so adding
    cases or changing tool sets doesn't masquerade as a regression.
    """
    ra = {r["id"]: r for r in a["cases"]}
    rb = {r["id"]: r for r in b["cases"]}
    common = sorted(set(ra) & set(rb))
    fixed = [i for i in common if not ra[i]["hit"] and rb[i]["hit"]]
    broke = [i for i in common if ra[i]["hit"] and not rb[i]["hit"]]

    def acc(rs):
        return sum(r["hit_rate"] for r in rs) / len(rs) if rs else 0.0

    acc_a = acc([ra[i] for i in common])
    acc_b = acc([rb[i] for i in common])
    ma, mb = a.get("meta", {}), b.get("meta", {})
    lines = [
        f"A: {ma.get('model', '?')} ({ma.get('timestamp', '?')})",
        f"B: {mb.get('model', '?')} ({mb.get('timestamp', '?')})",
        f"common cases {len(common)}",
        f"accuracy {acc_a:.1%} -> {acc_b:.1%}  ({(acc_b - acc_a) * 100:+.1f} pts)",
    ]
    if ma.get("cases_sha") != mb.get("cases_sha"):
        lines.append("note: the case file changed between runs")
    if ma.get("prompt_sha") != mb.get("prompt_sha"):
        lines.append("note: the system prompt changed between runs")
    la = a["summary"]["latency_s"]["median"]
    lb = b["summary"]["latency_s"]["median"]
    lines.append(f"median latency {la}s -> {lb}s")

    def first_got(r):
        return r["answers"][0]["got"]

    if fixed:
        lines.append("")
        lines.append(f"fixed ({len(fixed)})")
        lines += [f"  {i:<5} {first_got(ra[i])} -> {first_got(rb[i])}" for i in fixed]
    if broke:
        lines.append("")
        lines.append(f"REGRESSED ({len(broke)})")
        lines += [f"  {i:<5} {first_got(ra[i])} -> {first_got(rb[i])}   "
                  f"(expected {'/'.join(rb[i]['expect'])})" for i in broke]
    if not fixed and not broke:
        lines.append("no case changed outcome")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _load_tools_and_prompt(mode: str, with_memory: bool):
    """The exact prompt and tool list the agent would send in this mode.

    Imported late so `--compare` and `--list` work without the agent's deps.
    MEMORY.md is left out by default: it changes as you use the agent, and a
    prompt that drifts between runs would make before/after diffs meaningless.
    """
    import agent
    from tools import TOOL_REGISTRY, TOOL_SCHEMAS
    prompt = agent.build_system_prompt() if with_memory else agent.BASE_PROMPT
    try:
        import modes
    except ImportError:
        if mode != "general":
            raise SystemExit("modes.py not found — cannot use --mode")
        return prompt, list(TOOL_SCHEMAS)
    view = modes.apply_mode(mode, prompt, TOOL_SCHEMAS, TOOL_REGISTRY)
    return view.prompt, view.schemas


def _meta(args, schemas, prompt, source: pathlib.Path, t0: float) -> dict:
    return {
        "model": args.model,
        "mode": args.mode,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "num_ctx": args.ctx,
        "think": args.think,
        "temperature": args.temperature,
        "seed": args.seed,
        "runs": args.runs,
        "tools": sorted(schema_index(schemas)),
        "cases_sha": file_sha(source),
        "prompt_sha": text_sha(prompt),
        "wall_total_s": round(time.time() - t0, 1),
    }


def _save(report: dict, args, suffix: str = "") -> None:
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_model = args.model.replace(":", "_").replace("/", "_")
    path = out / f"{stamp}-{safe_model}-{args.mode}{suffix}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")


def _main_chains(args) -> int:
    source = pathlib.Path(args.chains)
    chains = load_chains(source)
    if args.only:
        wanted = {w.strip() for w in args.only.split(",") if w.strip()}
        chains = [c for c in chains if c["id"] in wanted or c["category"] in wanted]
    if args.list:
        for c in chains:
            print(f"{c['id']:<5} {c['category']:<14} "
                  f"{' -> '.join(s['expect'][0] for s in c['steps']):<40} {c['prompt']}")
        return 0
    prompt, schemas = _load_tools_and_prompt(args.mode, args.with_memory)
    temperature = None if args.temperature == "model" else float(args.temperature)
    chat = make_ollama_chat(args.model, args.ctx, args.think, temperature, args.seed, url=args.url)
    print(f"{len(chains)} chains, model {args.model}, mode {args.mode}, {len(schemas)} tools")
    t0 = time.time()
    report = run_chains(chains, chat, schemas, prompt)
    report["meta"] = _meta(args, schemas, prompt, source, t0)
    print(format_chain_summary(report))
    _save(report, args, "-chains")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Tool-selection evals for the local agent.")
    p.add_argument("--model", default="qwen3:8b")
    p.add_argument("--url", default=OLLAMA_URL, help="Ollama /api/chat endpoint")
    p.add_argument("--mode", default="general", help="tool subset from modes.py (default: general)")
    p.add_argument("--runs", type=int, default=1, help="repeat each case N times")
    p.add_argument("--ctx", type=int, default=16384)
    p.add_argument("--think", action="store_true", help="enable reasoning mode")
    p.add_argument("--temperature", default="0",
                   help="sampling temperature, or 'model' for the model's own default")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--only", default="", help="comma-separated case ids or categories")
    p.add_argument("--cases", default=str(CASES_FILE))
    p.add_argument("--with-memory", action="store_true",
                   help="include MEMORY.md in the prompt (less reproducible)")
    p.add_argument("--out", default=str(RESULTS_DIR))
    p.add_argument("--compare", nargs=2, metavar=("A", "B"), help="diff two saved reports")
    p.add_argument("--list", action="store_true", help="print the cases and exit")
    p.add_argument("--chains", nargs="?", const=str(CHAINS_FILE), default=None,
                   metavar="FILE", help="score multi-step chains instead (default evals/chains.jsonl)")
    args = p.parse_args(argv)

    if args.compare:
        a, b = (json.loads(pathlib.Path(x).read_text()) for x in args.compare)
        print(compare(a, b))
        return 0

    if args.chains:
        return _main_chains(args)

    cases = load_cases(pathlib.Path(args.cases))
    if args.only:
        wanted = {w.strip() for w in args.only.split(",") if w.strip()}
        cases = [c for c in cases if c["id"] in wanted or c["category"] in wanted]
    if args.list:
        for c in cases:
            print(f"{c['id']:<5} {c['category']:<10} {'/'.join(c['expect']):<28} {c['prompt']}")
        return 0

    prompt, schemas = _load_tools_and_prompt(args.mode, args.with_memory)
    temperature = None if args.temperature == "model" else float(args.temperature)
    chat = make_ollama_chat(args.model, args.ctx, args.think, temperature, args.seed, url=args.url)

    print(f"{len(cases)} cases, model {args.model}, mode {args.mode}, "
          f"{len(schemas)} tools, runs {args.runs}")
    t0 = time.time()
    report = run(cases, chat, schemas, prompt, runs=args.runs)
    report["meta"] = _meta(args, schemas, prompt, pathlib.Path(args.cases), t0)
    print(format_summary(report))
    _save(report, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
