# 3. Evals before model or prompt changes

Status: accepted

## Context

Tool selection, not tool execution, is this agent's weak point. Changes to
the model, tool descriptions or modes were previously judged by trying a few
prompts by hand, which cannot show a regression on cases nobody re-tried.

## Decision

No change to the model, a tool description, or a mode ships without an eval
number from before and after it (`python3 -m evals.run_evals`, compared
with `--compare`).

The harness is deliberately narrow:

- **Single step.** Each case scores the model's first action only: which
  tool (or none), and whether its arguments are valid.
- **Never executes tools.** Nothing touches the network beyond local Ollama,
  the workspace or the sandbox, so a run is safe to repeat.
- **Deterministic by default.** Temperature 0, seed 42, one run. The agent itself samples at the
  model's default temperature; `--temperature model --runs 3` measures that.
- **Results are committed** as JSON under `evals/results/`, so every number
  quoted elsewhere can be traced to a file.
- **Never tune against `evals/holdout.jsonl`.** Run it only to confirm a
  change that was tuned on `cases.jsonl`.

## Baseline

qwen3:8b, general mode, 14 tools: 95.5% accuracy over 44 scored cases (4
skipped), 100% argument validity, median latency 1.59 s
(`evals/results/20261015-033828-qwen3_8b-general.json`). Its misses:
`write_note → search_papers` and `fetch_url → none`.

The Qwen3.5-9B comparison is pending an Ollama upgrade (0.18.2 cannot pull
it). The default model does not change until it runs.

## Limits

- **Single-call accuracy overstates multi-turn skill.** A model can pick the
  right first tool and still lose the thread on step two or three. Multi-step
  chain evals with scripted tool results are the next step (task queue,
  "Multi-step evals").
- **One run at temperature 0 is one sample.** A one-case difference between
  two runs (about 2.3 points over 44 cases) is within noise; treat it as a tie
  unless repeated runs agree.
- **44 cases is a small set.** Add a case to `evals/cases.jsonl` whenever the
  agent picks wrong in real use.
