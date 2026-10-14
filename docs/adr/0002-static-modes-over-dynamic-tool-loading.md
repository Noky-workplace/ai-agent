# 2. Static project modes over dynamic tool loading

Status: accepted

## Context

Every tool schema goes to the model on every request. Small models pick
tools worse as the list grows and as tools look alike, so v7 needed a way to
add `get_prices` for finance work without widening every session.

Two options:

- **Dynamic loading:** the model searches for a tool, then it is loaded for
  the next call. That costs an extra model round-trip per turn and adds a
  second decision an 8B model can get wrong.
- **Static modes:** the user picks a project once (`--mode`, `/mode`, or a
  web UI button), which fixes a tool subset and appends a skill file to the
  system prompt for the whole session.

## Decision

Static modes (`modes.py`). `general` is the v6 tool set unchanged; `finance`
and `math3332` each send 12 tools.

The model receives only the mode's schemas, and `execute_tool` dispatches
only through the mode's registry. A caller that passes no registry gets
general mode's, never the raw global registry. Hiding a schema alone would
not be a boundary: a model can still emit a hidden tool's name.

## Evidence

Measured on this Mac with qwen3:8b, one run each at temperature 0:

- **The prefix cache works, so schema count is cheap per turn.** Cold
  prefill of the 1,792-token prompt took 10.631 s; a new turn on the same
  prefix took 0.221 s, 2% of cold
  (`evals/results/20261015-prefix-cache-qwen3_8b-general.txt`). Loading tools
  dynamically would save little and cost a round-trip.
- **General vs finance:** 95.5% vs 93.2% accuracy over 44 scored cases each
  (`evals/results/20261015-033828-qwen3_8b-general.json`,
  `evals/results/20261015-034040-qwen3_8b-finance.json`). The rows skip
  different cases (tools absent from the mode). On the 40 cases both modes
  share, each scores 38/40, so dropping two tools did not change accuracy.
  Finance's lower total comes from its own market cases: `get_prices`
  scored 3 of 4.

## Consequences

- A mode switch rebuilds the prompt and clears history, because the prefix
  changes anyway.
- Finance mode's subset omits `extract_archive`, `edit_file` and
  `delete_file`. Revisit if finance work needs them.
- Revisit dynamic loading only if the tool count passes roughly 25 or the
  prefix cache stops holding (re-run `python3 -m evals.prefix_cache` after
  Ollama upgrades).
