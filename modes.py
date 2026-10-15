"""
v7 — Project modes: a static tool subset plus a skill file, per project.

    general    everything except the market tool (identical to v6)
    finance    research + data + compute tools, and get_prices
    math3332   compute + file + notes tools for TA prep

WHY STATIC SUBSETS AND NOT DYNAMIC TOOL LOADING

Two things push a small model's tool choice wrong: too many tools, and
tools that look alike. Dynamic loading ("search for a tool, then load it")
fixes the count but costs an extra model round-trip on every turn, and adds
a second decision an 8B model can get wrong. At 14 tools the savings it
offers don't exist — the schema prefix is cached after the first request.

A mode picks the subset once, when you choose the project, not per turn.
The model sees 12 tools instead of 14, without wasting an iteration or a
cache miss. That's the same trade the research
pointed at ("mode-based static grouping"), and it's measurable: run
`python3 -m evals.run_evals --mode finance` and compare against general.

SKILLS

Each mode can append a markdown file from skills/ to the system prompt —
the domain rules a generic prompt can't carry (backtests must not look
ahead; every TA solution is checked by running it). Loaded once per session,
so it is part of the cached prefix, not a per-turn cost.

SECURITY NOTE FOR INTEGRATION

A mode must filter the REGISTRY as well as the schemas. If the agent still
dispatches through the full registry, a model that hallucinates the name of
a hidden tool gets it executed anyway. ModeView.registry is the filtered one
— dispatch through that.
"""

import dataclasses
import pathlib

SKILLS_DIR = pathlib.Path(__file__).resolve().parent / "skills"


@dataclasses.dataclass(frozen=True)
class Mode:
    name: str
    description: str
    include: tuple[str, ...] = ()   # if set, only these tools
    exclude: tuple[str, ...] = ()   # if set, everything except these
    skill: str = ""                 # file in skills/, appended to the prompt


@dataclasses.dataclass
class ModeView:
    """What the agent should actually send and dispatch through in a mode."""
    name: str
    prompt: str
    schemas: list
    registry: dict
    missing: list   # tools the mode asked for that this checkout doesn't have


MODES = {
    "general": Mode(
        name="general",
        description="Everything except market data.",
        exclude=("get_prices",),
    ),
    "finance": Mode(
        name="finance",
        description="Market research and backtesting.",
        include=(
            "get_current_time", "web_search", "fetch_url", "search_papers",
            "get_prices", "run_python", "read_file", "list_files", "write_file",
            "write_note", "search_notes", "remember",
        ),
        skill="finance.md",
    ),
    "math3332": Mode(
        name="math3332",
        description="MATH 3332 teaching-assistant prep.",
        include=(
            "get_current_time", "run_python", "read_file", "list_files",
            "write_file", "edit_file", "write_note", "search_notes",
            "search_papers", "web_search", "fetch_url", "remember",
        ),
        skill="math3332.md",
    ),
}


def extra_schemas() -> list:
    """Tool schemas that exist only through modes (not merged into tools.py).

    get_prices lives here rather than in tools.py so general mode is exactly
    v6's tool set, and so this file doesn't have to edit tools.py.
    """
    try:
        from market import MARKET_TOOL_SCHEMAS
        return list(MARKET_TOOL_SCHEMAS)
    except ImportError:
        return []


def _extra_registry() -> dict:
    try:
        from market import MARKET_TOOL_REGISTRY
        return dict(MARKET_TOOL_REGISTRY)
    except ImportError:
        return {}


def load_skill(filename: str) -> str:
    if not filename:
        return ""
    path = SKILLS_DIR / filename
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def apply_mode(name: str, base_prompt: str, schemas: list, registry: dict,
               with_skill: bool = True) -> ModeView:
    """Return the prompt, schemas and registry for a mode.

    Tools the mode lists but this checkout lacks are dropped and reported in
    `missing`, so a mode file can name a tool before it exists without
    crashing older code. Schemas and registry are reconciled both ways: a
    schema with no callable behind it is hidden (and reported), and a
    registry key with no schema is dropped — which also scrubs junk like
    the v5 'type' phantom.
    """
    mode = MODES.get(name)
    if mode is None:
        raise ValueError(f"unknown mode {name!r}; choose from {', '.join(MODES)}")

    all_schemas = list(schemas)
    names = {s["function"]["name"] for s in all_schemas}
    for s in extra_schemas():
        if s["function"]["name"] not in names:
            all_schemas.append(s)
            names.add(s["function"]["name"])
    all_registry = {**registry, **_extra_registry()}

    if mode.include:
        keep = [n for n in mode.include if n in names]
        missing = [n for n in mode.include if n not in names]
        order = {n: i for i, n in enumerate(mode.include)}
        chosen = sorted((s for s in all_schemas if s["function"]["name"] in keep),
                        key=lambda s: order[s["function"]["name"]])
    else:
        missing = []
        chosen = [s for s in all_schemas if s["function"]["name"] not in mode.exclude]

    # Never show the model a tool that can't run. A schema with no registry
    # entry is exactly how fetch_url was dead for a month in v5: the model
    # saw it, called it, and got "no tool named 'fetch_url'".
    unregistered = [s["function"]["name"] for s in chosen
                    if not callable(all_registry.get(s["function"]["name"]))]
    chosen = [s for s in chosen if s["function"]["name"] not in unregistered]
    missing += [f"{n} (schema but no implementation)" for n in unregistered]

    chosen_names = {s["function"]["name"] for s in chosen}
    reg = {k: v for k, v in all_registry.items() if k in chosen_names}

    # with_skill=False keeps the tool subset but drops the skill text, so an
    # eval can tell which of the two changes the model's behaviour.
    skill = load_skill(mode.skill) if with_skill else ""
    prompt = base_prompt
    if skill:
        prompt = f"{base_prompt}\n\n--- Mode: {mode.name} ---\n{skill}"
    return ModeView(name=mode.name, prompt=prompt, schemas=chosen,
                    registry=reg, missing=missing)
