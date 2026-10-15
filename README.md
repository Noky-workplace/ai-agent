# Local Agent — v7

A fully-local personal assistant built from scratch (no agent frameworks).
Runs on a MacBook Air M4 via Ollama.

- **v0** chat loop · **v1** tool calling · **v2** file memory
- **v3** academic paper search + Python execution · **v4** `fetch_url`
- **v5** safe file I/O + Seatbelt sandbox · **v6** web UI, persistent `sandbox_io/`, tests
- **v7** eval harness, project modes, memory fixes

## Results

Every design decision in this repo was made by measuring it. Tool selection is
scored by an eval harness that never executes a tool, the numbers below come
from committed result files under `evals/results/`, and a decision that didn't
survive a holdout set was reverted. Model: `qwen3:8b` (Q4_K_M) via Ollama on a
16 GB MacBook Air M4.

### How it's measured

- **Single-step:** 44 cases (`evals/cases.jsonl`). Does the model's first action
  pick the right tool, or correctly none, with schema-valid arguments?
- **Multi-step:** 10–13 chains (`evals/chains.jsonl`) with scripted tool results
  fed back, scored per step. Checks include "copied old_string from the file" and
  "opened the exact CSV path it was given".
- **Holdouts, never tuned on:** 22 cases (`holdout.jsonl`) and 18 harder ones,
  10 of them in Cantonese (`holdout_hard.jsonl`), written apart from the tuning.
- Temperature 0, fixed seed. Three sampled runs at the model's own temperature
  chose the same tool on every case, so a one-case change reflects the change
  being tested, not noise.

### Headline numbers

| What | Result | Source |
|---|---|---|
| Tool selection, general mode | 95.5% → 97.7% after tightening tool descriptions; 100% after one flawed case was corrected (that run on Ollama 0.40.4) | `…033828…general.json`, `…045003…general.json`, `…230843…general.json` |
| Argument validity | 100% in every run | all result files |
| Holdout, never tuned on | 100% (21 of 22 cases scored in general mode), before and after the description change | `…holdout-pre5.json`, `…holdout-head.json` |
| Hard holdout, incl. 10 Cantonese | 100% general mode (17 of 18 scored); Cantonese 9/9, English 8/8 | `…general-holdout-hard.json` |
| Multi-step chains, general | 80% → 100% after the fixes below | `…051517…chains.json`, `…063843…chains.json` |
| Prompt cache (system prompt + 14 schemas) | a new turn prefills in 0.221 s vs 10.631 s cold (2%) | `…prefix-cache-qwen3_8b-general.txt` |
| Notes search, 15 queries | grep alone 33% (synonym queries 0 of 10) → grep + embedding fallback 100%, right note first 14 of 15 | `…notes-retrieval.json` |

### Decisions the numbers made

| Question | Evidence | Decision |
|---|---|---|
| Load tools on demand, or keep all 14 loaded? | The cached prefix makes the schemas cost 2% of a cold prefill per turn | Keep all loaded; add static per-project modes instead (ADR 0002) |
| Shorter tool descriptions? | Schema tokens about −18% (estimated, commit `3ca76c3`), holdout unchanged at 100%; one shorter `run_python` description lost a case | Kept, except `run_python`, which stays longer |
| Why does finance mode skip tools? | Ablation: with its skill text 83.3% on the hard holdout, without it 100% | The skill text is the cause; a just-in-time variant scored 88.9% and was reverted. Finance mode now runs without the skill text |
| Upgrade to Qwen3.5-9B? | Accuracy a tie (general: 8B 100%, 9B 97.7%; finance: 8B 95.5%, 9B 97.7%). 9B is slower (median 1.55 → 1.78 s general, 1.86 → 3.10 s finance) and peaks at all 16 GB of RAM | Stay on `qwen3:8b` |

### Bugs the evals and tests caught

- **A safety bug in an error message.** The edit tool's "text appears twice"
  error offered `replace_all=true` as an equal option, and the model took it,
  which would rewrite every match. Reworded; the chain now passes.
- **Five memory bugs**, found by writing `test_memory.py`. Two hit Chinese text
  specifically: Chinese note topics were rejected outright, and duplicate
  detection never fired on Chinese facts (now handled with character bigrams).
- **A tool that was silently dead for a month.** A registry typo meant the model
  saw `fetch_url` but nothing could run it. Modes now refuse to show any tool
  without an implementation, and a test pins it.

### Limits

- One model, small sets: 44, 22 and 18 cases, plus 10–13 chains. One case is
  about 2 points.
- Tool choice is measured; answer quality isn't. A backtest with lookahead bias
  would pass every eval here.
- Built with Claude and Claude Code as collaborators (see `Co-Authored-By` in
  the log). The decisions, reviews and measurements are traceable commit by
  commit.

## Setup

```bash
ollama pull qwen3:8b
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
mkdir -p notes
.venv/bin/python agent.py          # terminal
.venv/bin/python server.py         # web UI at http://127.0.0.1:8765
```

Run it from the directory you want it to work in: that directory is the
**workspace**, and every file tool is confined to it.

| Flag | Effect |
|---|---|
| `--model` | Ollama model (default `qwen3:8b`) |
| `--ctx N` | context window in tokens |
| `--think` | Qwen3 reasoning mode (much slower) |
| `--plan` | plan mode: writes, edits, deletes and extracts are refused |
| `--mode` | `general` (default), `finance` or `math3332`; see v7 below |
| `--time`, `--quiet` | per-call timing / hide tool traces (terminal only) |

Commands: `/clear` `/mode <name>` `/history` `/memory` `/context` `/undo` `/exit`
(the web UI supports `/clear` and `/undo`; its two welcome buttons switch mode).

## Tools (14)

Every tool schema is sent to the model on **every request**, which costs
context and, more importantly, **selection accuracy**: an 8B model in this
project once failed to pick `web_search` out of four options. The rule is
**fewer, more distinct tools beat more, narrower ones** — e.g. there is no
`calculator` (that's `run_python`), and arXiv and Semantic Scholar share one
`search_papers` tool with a `source` parameter.

| Tool | Purpose |
|------|---------|
| `get_current_time` | date/time |
| `web_search` | current information (DuckDuckGo) |
| `fetch_url` | full text of one page, after `web_search` |
| `search_papers` | arXiv or Semantic Scholar |
| `remember` | short durable facts → `MEMORY.md` |
| `write_note` / `search_notes` | longer content in `notes/`, literal grep |
| `read_file` / `list_files` | read the workspace |
| `write_file` / `edit_file` / `delete_file` | change the workspace (approval) |
| `run_python` | sandboxed Python, cwd `sandbox_io/` (approval) |
| `extract_archive` | unpack zip/tar into `sandbox_io/<name>/` (approval) |

## v7: evals, modes, memory fixes

### Evals (`evals/`)

`python3 -m evals.run_evals` sends each case in `evals/cases.jsonl` to the
model once and scores its **first action**: which tool it calls (or none) and
whether the arguments are valid. No tool is ever executed. Temperature 0,
seed 42. Results are saved under `evals/results/`.

Baseline, qwen3:8b on this Mac (one run each):

| Mode | Tools | Cases scored | Accuracy | Arg validity | Median latency | Decode |
|---|---|---|---|---|---|---|
| general | 14 | 44 (4 skipped) | 95.5% | 100% | 1.59 s | 19.8 tok/s |
| finance | 12 | 44 (4 skipped) | 93.2% | 100% | 1.67 s | 18.9 tok/s |

Misses: general `write_note → search_papers`, `fetch_url → none`; finance
`web_search → none`, `write_note → none`, `get_prices → get_current_time`.
A case is skipped when its tool is not in the mode, so the two rows score
different sets of 44; on the 40 cases both modes share, each scores 38/40.
Sources: `evals/results/20261015-033828-qwen3_8b-general.json`,
`evals/results/20261015-034040-qwen3_8b-finance.json`.

**Prefix cache: cached.** `python3 -m evals.prefix_cache` measured 10.631 s
to prefill the 1,792-token prompt cold, 0.221 s for a new turn on the same
prefix (2%). So keeping every tool schema loaded costs once per session, not
per turn (`evals/results/20261015-prefix-cache-qwen3_8b-general.txt`).

**Qwen3.5-9B comparison: done** (Ollama 0.40.4). Accuracy tied, 9B was slower
and peaked at all 16 GB of RAM, so the default stays `qwen3:8b`; see Results
(`evals/results/20261015-compare-qwen3_8b-vs-qwen3.5_9b.txt`).

**Real-use traces.** Each turn appends one line (prompt, tool names, argument names, outcomes; never values or results) to the git-ignored `.agent-cache/traces.jsonl`; `--no-trace` turns it off.
`python3 -m evals.mine_traces` prints errored, denied, stuck or suspicious no-tool turns as draft cases with an empty `expect`, for you to check before adding any to `cases.jsonl`.

### Modes (`modes.py`, `skills/`)

A mode is a fixed tool subset plus a skill file appended to the system prompt,
chosen once per session (`--mode`, `/mode`, or a web UI button):

| Mode | Tools | Adds |
|---|---|---|
| `general` | the 14 above | nothing (identical to v6) |
| `finance` | 12 | `get_prices` (Yahoo Finance via optional `yfinance`, CSV into `sandbox_io/data/`); no skill text, since it made the model skip tools (`skills/finance.md` kept, off) |
| `math3332` | 12 | TA-prep rules |

The model is sent only the mode's schemas, and tool calls dispatch only
through the mode's registry, so a hidden tool's name is refused even if the
model emits it (`test_agent.py`). See ADR 0002.

### Memory fixes (`memory.py`)

Five bugs, each pinned by a regression test: the duplicate check matched
header text and longer facts; the `_(updated …)_` marker leaked dates into
comparisons; `replaces=` overwrote the first of several matches (now refused
unless exactly one matches); Chinese note topics were stripped to nothing;
and Chinese facts had no "words", so duplicates were never caught (CJK text
now uses character bigrams).

## File tools (`files.py`)

Paths and content come from the model, and the model can be steered by
anything it reads, so every argument is treated as hostile.

- **Containment.** Paths are resolved with symlinks followed *before* the
  workspace prefix check, so a link pointing outside is caught.
- **Denylist.** `.env*`, keys, `.ssh`, `.git`, `.venv`, `MEMORY.md` and friends
  are never readable or writable, compared case-folded and NFC-normalised
  (APFS treats `.ENV` and `.env` as the same file).
- **Approval.** Reads are free. Every write, edit and delete shows a diff and
  asks. `--plan` disables mutation outright.
- **Recoverable.** Deletes go to `.trash/`; writes are atomic and re-read to
  verify; every mutation is checkpointed to a shadow git repo (`.agent-git/`)
  first, so `/undo` rolls it back.

## Code execution (`sandbox.py`, `sandbox.sb`)

`run_python` runs behind **a human approval prompt** and, on macOS, inside a
deny-default **Seatbelt** profile (`sandbox-exec`):

| Threat | Handled? |
|---|---|
| infinite loop / crash | yes — separate process, 15s timeout |
| runaway output floods context | yes — truncated |
| reading `~/.ssh/...` | yes — kernel denies reads outside the interpreter and `sandbox_io/` |
| writing `agent.py`, `MEMORY.md`, `notes/` | yes — `sandbox_io/` is the only writable path |
| network requests | yes — `(deny network*)` |
| memory/CPU exhaustion | **no** |
| a bug in the profile | **no** — Seatbelt profiles have been escaped in shipped tools |

The approval prompt stays the primary control. Off macOS there is no
sandbox, and the tool result says so. Set `REQUIRE_APPROVAL = False` only
inside a disposable container.

### `sandbox_io/`

The sandbox's cwd and only writable path, persisted between calls and
visible to the file tools, so data moves without any new mechanism:

- **in:** `write_file("sandbox_io/data.csv", ...)`, `extract_archive("data.zip")`,
  or drop files there by hand
- **compute:** `run_python` reads and writes relative paths freely
- **out:** `read_file("sandbox_io/result.json")`

It is git-ignored and excluded from checkpoints, so `/undo` does not restore
it. See [`docs/adr/0001-persistent-sandbox-io.md`](docs/adr/0001-persistent-sandbox-io.md).

## Web UI (`server.py`)

A single-page chat on `127.0.0.1:8765` (localhost only — never change the
host). Approval prompts stay **in the terminal**: the browser shows "waiting
for approval" and the server raises Terminal.app until you answer.

## Try this

```
you> find recent papers on retrieval augmented generation
you> what's the compound interest on 5000 at 4.5% for 7 years?
you> unpack data.zip and tell me the mean of the price column
you> find papers on quantization and save a summary to my notes
```

Multi-step tool chains are where small models most often lose the thread.

## Files

```
agent.py         ReAct loop + CLI
server.py        web UI
tools.py         registry: merges all tool modules
memory.py        MEMORY.md + notes/
research.py      arXiv + Semantic Scholar
fetch.py         fetch_url
files.py         workspace file tools, containment, checkpoints
sandbox.py       run_python + extract_archive
sandbox.sb       Seatbelt profile
modes.py         project modes: tool subsets + skills/
market.py        get_prices (finance mode)
skills/          per-mode prompt rules
evals/           tool-selection eval harness + results/
bench.py         latency profiling (why is the agent slow?)
test_*.py        unit tests (files, sandbox, memory, modes, market, evals, agent, server)
```

Adding a capability = one module exporting `*_TOOL_SCHEMAS` and
`*_TOOL_REGISTRY`, plus two lines in `tools.py`. Check the tool budget first.

```bash
.venv/bin/python -m unittest test_files test_sandbox test_memory test_modes test_market test_evals test_agent test_server
```

## Known limitations

- **Tool selection is the weak point**, not tool execution. Qwen3.5-9B tied
  on accuracy but was slower and used all 16 GB; see Results.
- **`search_notes` greps first.** Only when grep finds nothing does it fall
  back to embedding similarity (top 3, labelled as semantic matches), so "CNN"
  can find "convolutional networks" (ADR 0004).
- **Semantic Scholar rate-limits anonymous traffic.** On HTTP 429, wait or
  use `source='arxiv'`.
- **Prompt injection is contained, not prevented.** A malicious page can
  still steer the model; the denylist, approvals, and sandbox limit what it
  can do with that.
