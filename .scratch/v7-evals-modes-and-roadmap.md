# v7 — evals, memory fixes, project modes, and what comes after

Status: open — built and tested overnight in a cloud clone, not yet on your Mac
Base: `ae8cc6c` (origin/main). Your 6 unpushed v6 commits are NOT in that base.
Delivery: `v7.mbox` — 6 commits (the last adds this brief), applied with one command (see Morning step 2).

---

## What's in v7.mbox

| Commit | What | Touches |
|---|---|---|
| test: cover memory.py | 20 tests pinning current memory behaviour | new file |
| fix(memory): five bugs | dedup/supersede fixes, incl. Chinese text; +12 regression tests | `memory.py` |
| feat: modes, skills, get_prices | `modes.py`, `market.py`, `skills/*.md`, 2 test files | new files only |
| feat: eval harness | `evals/`, `test_evals.py` | new files only |
| fix(evals): all-skipped summary | crash fix found in final review | `evals/` |
| docs: v7 brief | this file | new file |

Only one existing file is edited (`memory.py`), so the patches apply on top of
your v6 commits. The one overlap risk: v6 deleted `MAX_NOTE_CHARS` from
`memory.py`. The memory fix doesn't touch that line; `git am -3` resolves the
offset with a 3-way merge. Verified by applying to a simulated v6 (see bottom).

Test totals after applying: files 24, memory 32, evals 32, market 15, modes 12,
plus v6's test_sandbox.py.

---

## Morning — you, ~15 minutes

1. **Push v6 first** — it's reviewed, it's yours, it shouldn't wait on v7.
   ```bash
   cd ~/projects/ai-agent
   git push
   ```
2. **Apply v7.** Put `v7.mbox` in the repo folder, then:
   ```bash
   git am -3 v7.mbox
   python3 -m unittest test_files test_memory test_evals test_market test_modes -v
   ```
   If `git am` stops on a conflict: `git am --abort` and hand this file to Claude Code.
3. **Optional dependency** for finance mode: `pip install yfinance` (add to requirements.txt).
4. **Is the prefix cache real?** (section 6 of the v6 brief, now one command; needs Ollama running)
   ```bash
   python3 -m evals.prefix_cache
   ```
   Want `new turn : CACHED`. If not: update Ollama (prompt-cache rework landed in v0.30.8,
   June 2026) and re-run. This decides whether keeping all tools loaded is still right.
5. **Baseline.** About 3–5 minutes:
   ```bash
   python3 -m evals.run_evals                       # qwen3:8b, general mode
   ```
   Commit the JSON it saves under `evals/results/` — the "before" number, which
   every later change is compared against.
6. **The model question.** Qwen3.5-9B is the current top pick for 16GB machines
   (vendor-reported BFCL-V4 66.1, τ²-bench 79.1 — the vendor's harness, not yours):
   ```bash
   ollama pull qwen3.5:9b
   python3 -m evals.run_evals --model qwen3.5:9b
   python3 -m evals.run_evals --compare evals/results/<qwen3>.json evals/results/<qwen3.5>.json
   ```
   Switch the default only if accuracy improves AND latency is acceptable. Check memory with
   `ollama ps` while it runs — 6.6–7.6GB of weights plus the KV cache at 16k context, on a 16GB
   machine also running a browser.

Not worth chasing: Ollama's MLX engine. It activates only on Macs with 32GB+; on 16GB it
silently falls back to the Metal path you already have.

---

## For Claude Code — integration (needs your v6 files, so it couldn't be done in the cloud)

### A. Modes in `agent.py`

- `--mode {general,finance,math3332}` flag, default `general`.
- After building the system prompt:
  `view = modes.apply_mode(args.mode, system_prompt, TOOL_SCHEMAS, TOOL_REGISTRY)`
  and use `view.prompt` as the system message.
- **Security requirement, not style:** `call_ollama` must send `view.schemas`, and
  `execute_tool` must dispatch through `view.registry` — NOT the global `TOOL_REGISTRY`.
  Otherwise a model that hallucinates a hidden tool's name gets it executed.
  Thread `schemas`/`registry` through `run_agent` as parameters defaulting to the globals.
- Banner: show mode, tool count, and `view.missing` if non-empty (it flags a schema with no
  implementation — the exact way `fetch_url` stayed dead for a month).
- `/mode <name>` command: rebuild the view and clear history (the prefix changes anyway).
- `/clear` must rebuild with the current mode, not fall back to general.

### B. Modes in `server.py`

- The two welcome buttons already exist. Have each one send its mode with the chat request:
  `{"text": ..., "mode": "finance"}`. If the mode differs from the current one, the server
  rebuilds the view and resets the conversation, then runs the turn.
- Show the current mode in the header next to model/status.
- Same requirement: the turn dispatches through the mode's registry.

### B2. Check one assumption in `market.py`

`get_prices` tells the model to open `data/<file>.csv` relative to run_python's working
directory, assuming v6 made that directory `sandbox_io/` (as the v6 brief specified). It
locates the folder via `sandbox.IO_DIR`, falling back to `<workspace>/sandbox_io`. Confirm
v6's sandbox.py matches; if the name differs, adjust `market._io_dir` and the hint in
`market._summary`, then run `test_market`.

### C. Docs

- README: a "v7" section — evals (with the baseline number), modes, the memory fixes.
- `docs/adr/0002-static-modes-over-dynamic-tool-loading.md` — why: per-turn tool search costs a
  round-trip and a second decision an 8B model can get wrong, and prefix caching already
  makes 14 tool schemas cheap. Evidence: the eval run comparing general vs finance.
- `docs/adr/0003-evals-before-model-changes.md` — single-step, never executes tools,
  deterministic by default; and its limit (single-call accuracy overstates multi-turn skill).
- `requirements.txt`: `yfinance` (optional; market.py imports it lazily).

### D. Commit and push

One commit per section, A–C. Run the full test suite before each. Push when green.

---

## Roadmap — after v7, ranked

Each item's value is measured with the harness, not assumed. Rule: **no change ships without
an eval number from before and after it.**

1. **Pick the model by evidence** (Morning step 6). Biggest single lever, ~10 minutes.
2. **Tune tool descriptions against the confusions list.** The harness prints the top
   "expected → got" pairs. Rewrite the losing descriptions (v6 section 6, now measurable), re-run, keep
   what improves. Add a case to `cases.jsonl` whenever the agent picks wrong in real use.
3. **Multi-step evals (v8).** Single-step accuracy is a floor. Next: scripted tool results, so
   the harness can play 2–3 steps without executing anything — e.g. does `web_search` lead to
   `fetch_url` on the right link? Does `get_prices` lead to `run_python` opening the right file?
   Reuses `first_action`/`check_args`; the new part is a fake tool table per case.
4. **Notes search that finds synonyms.** Known gap since v2: a grep for "CNN" misses "convolutional
   networks". Keep grep as the first try (inspectable, ADR 001), and fall back to embeddings
   from Ollama (`nomic-embed-text`) only when it finds nothing. Measure with a small
   notes-retrieval eval before and after.
5. **Stream the final answer** in the web UI. Pure UX; at 15 tok/s a 200-token answer is 13 s of
   blank screen. Tool calls stay non-streamed.
6. **Planning tool** — only if multi-step evals show the agent losing track mid-task. Research
   says todo lists help small models, but it's tool #15; earn it with a number first.

**Not worth it on this machine** (from the September research, still true):
LLM compaction (~100 s per pass on an 8B model, and small-model summaries drop tool details);
dynamic tool loading under ~25 tools; MLX (needs 32GB); MCP client (adds tools, hurts selection).

---

## Next time: letting the cloud session hand work to Claude Code directly

Tonight the cloud session couldn't reach your Mac — `ListAgents` showed nothing. Per the docs,
a Claude Code session becomes reachable from your other sessions when you run it with
Remote Control:

```bash
claude --remote-control          # or, inside a running session: /remote-control
```

Requirements: claude.ai login (not an API key), Claude Code ≥ v2.1.225 for cross-machine
messages, and the Mac awake — if it sleeps, messages queue until it reconnects. Untested
from a claude.ai cloud session; try it once while you're at the keyboard before relying on it
overnight. A message from another session can't approve anything on your Mac: Claude Code's
permission prompts still fire there.

---

## Verification done before this was handed over

- All five suites pass in the cloud clone (Python 3.13): 24 + 32 + 32 + 15 + 12.
- The eval CLI ran end to end against a stand-in Ollama server: run, `--mode finance`,
  `--compare`, `--list`, and `prefix_cache.py`. Real-model numbers can only come from your Mac.
- `v7.mbox` applied cleanly with `git am -3` onto a fresh clone of `ae8cc6c` with v6's
  `memory.py` change (deleted `MAX_NOTE_CHARS`) and `tools.py` fix simulated, and every suite
  passed there afterwards.
- `get_prices` is tested with a fake yfinance returning a real pandas DataFrame. It has never
  hit Yahoo — the cloud sandbox can't reach it. First real call is yours.
