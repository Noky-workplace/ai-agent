# Local Agent — v7

A fully-local personal assistant built from scratch (no agent frameworks).
Runs on a MacBook Air M4 via Ollama.

- **v0** chat loop · **v1** tool calling · **v2** file memory
- **v3** academic paper search + Python execution · **v4** `fetch_url`
- **v5** safe file I/O + Seatbelt sandbox · **v6** web UI, persistent `sandbox_io/`, tests
- **v7** eval harness, project modes, memory fixes

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

**Qwen3.5-9B comparison: pending.** Ollama 0.18.2 cannot pull `qwen3.5:9b`
(HTTP 412, needs a newer Ollama). The default model stays `qwen3:8b` until
that run exists.

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

- **Tool selection is the weak point**, not tool execution. Larger models
  (`qwen3:14b`) help if you have the RAM.
- **`search_notes` is literal matching.** "CNN" won't find "convolutional
  networks". Deliberate tradeoff.
- **Semantic Scholar rate-limits anonymous traffic.** On HTTP 429, wait or
  use `source='arxiv'`.
- **Prompt injection is contained, not prevented.** A malicious page can
  still steer the model; the denylist, approvals, and sandbox limit what it
  can do with that.
