# Local Agent — v3: Research & Code Execution

A fully-local personal assistant built from scratch (no agent frameworks).
Runs on a MacBook Air M4 via Ollama.

- **v0** chat loop · **v1** tool calling · **v2** file memory
- **v3** academic paper search + Python execution

## Setup

```bash
ollama pull qwen3:8b
pip3 install ddgs
mkdir -p notes
python3 agent.py
```

No new dependencies in v3 — arXiv and Semantic Scholar are stdlib `urllib`.

## The tool budget problem

Every tool schema is sent to the model on **every request**. Two costs:
context (~50–150 tokens each) and, more importantly, **selection accuracy**.
An 8B model in this project already failed to pick `web_search` when there
were only four options. Small models degrade further past ~8–10 tools.

So v3 added capability while holding the count to **8**, via two decisions:

**1. Dropped `calculator`.** `run_python` does everything it did. Two
overlapping math tools is precisely the ambiguity that causes bad picks.

**2. One `search_papers` tool, not two.** arXiv and Semantic Scholar are one
tool with a `source` parameter. Picking a parameter from a two-item enum is
an easier decision than picking between two near-identical tools.

The general principle: **fewer, more distinct tools beat more, narrower ones.**

| Tool | Purpose |
|------|---------|
| `get_current_time` | date/time |
| `web_search` | current information (DuckDuckGo) |
| `search_papers` | arXiv or Semantic Scholar |
| `run_python` | calculations, data processing, code testing |
| `read_file` | project files and notes |
| `remember` | durable facts to MEMORY.md |
| `write_note` | longer content to notes/ |
| `search_notes` | grep notes |

## Code execution: isolation, not a sandbox

`run_python` runs code in a subprocess with a fresh temp directory, a 15s
timeout, output truncation, and **a human approval prompt**.

Be clear about what that does and does not buy you:

| Threat | Handled? |
|---|---|
| infinite loop hangs the agent | yes — timeout kills it |
| crash takes down the agent | yes — separate process |
| relative file writes hit your project | yes — temp cwd |
| runaway output floods context | yes — truncated |
| **reading `~/.ssh/id_ed25519`** | **no** — absolute paths work |
| **network requests** | **no** |
| **memory/CPU exhaustion** | **no** |

Real isolation needs an OS boundary: Docker with `--network=none` and a
read-only mount, or a VM. Until then, **the approval prompt is the actual
security control** — read the code before typing `y`.

Set `REQUIRE_APPROVAL = False` in `sandbox.py` only inside a disposable
container.

## Try this

```
you> find recent papers on retrieval augmented generation
you> search semantic scholar for attention is all you need, how many citations?
you> what's the compound interest on 5000 at 4.5% for 7 years?
you> write a function to check if a string is a palindrome and test it
you> find papers on quantization and save a summary to my notes
```

The last one is the real test: `search_papers` then `write_note` in one turn.
Multi-step tool chains are where small models most often lose the thread.

## Files

```
agent.py      ReAct loop + CLI
tools.py      registry: merges all tool modules
memory.py     MEMORY.md + notes/  (v2)
research.py   arXiv + Semantic Scholar  (v3)
sandbox.py    Python execution  (v3)
```

Adding a capability = one new module exporting `*_TOOL_SCHEMAS` and
`*_TOOL_REGISTRY`, plus two lines in `tools.py`. But check the tool budget
first — at 8, the next addition should probably replace something.

## Known limitations

- **Tool selection is the weak point**, not tool execution. If the agent
  answers without searching, that's the 8B model being overconfident. Larger
  models (`qwen3:14b`) improve this if you have the RAM.
- **`search_notes` is literal matching.** "CNN" won't find a note about
  "convolutional networks". Deliberate tradeoff — see ADR 001.
- **Semantic Scholar rate-limits anonymous traffic.** On HTTP 429, wait or
  use `source='arxiv'`.
- **No prompt-injection defense yet.** A malicious page from `web_search`
  could try to influence the agent. This is now more serious than in v2,
  because the agent can write files *and* execute code. **v4's job.**

## Roadmap

- ~~v0 chat~~ · ~~v1 tools~~ · ~~v2 memory~~ · ~~v3 research + code~~
- **v4** — evals + guardrails (prompt-injection defense, audit log,
  tool-selection accuracy measurement)
- **v5** — optional web UI (FastAPI)
- **v6** — optional QLoRA fine-tune (needs a cloud GPU)
- *deferred*: FinTech/quant tools (yfinance) — skipped to protect the tool
  budget; add when routing is solved in v4.
