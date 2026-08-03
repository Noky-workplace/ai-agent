"""
v3 — Tool registry.

A "tool" is two things bolted together:
  1. A normal Python function that does something.
  2. A JSON schema describing it, which we send to the model so it knows
     the tool exists, what it does, and what arguments it takes.

The model never runs code. It only ever *asks* us to run a tool by emitting
JSON like {"name": "calculator", "arguments": {"expression": "2+2"}}.
We validate that request, run the function ourselves, and hand back the
result. That gap — model proposes, our code disposes — is the entire
security model of an agent. Guard it carefully.

Keep tool descriptions short and concrete: small models follow them better,
and every schema costs ~50-150 tokens of context on every single turn.
"""

import datetime
import pathlib

# Only files under this directory can be read. Anything else is refused.
# This is "least privilege": the tool physically cannot do what it isn't for.
WORKSPACE = pathlib.Path.cwd().resolve()

MAX_FILE_CHARS = 4000
MAX_SEARCH_RESULTS = 5


# --------------------------------------------------------------------------
# Tool implementations
# --------------------------------------------------------------------------

def get_current_time() -> str:
    """Local date/time. Fixes the 'I can't access real-time data' gap."""
    now = datetime.datetime.now().astimezone()
    return now.strftime("%A, %d %B %Y, %H:%M:%S %Z")


def web_search(query: str) -> str:
    """Search the web via DuckDuckGo. No API key required."""
    try:
        from ddgs import DDGS
    except ImportError:
        return "Error: ddgs not installed. Run: pip install ddgs"

    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=MAX_SEARCH_RESULTS))
    except Exception as exc:
        return f"Error: search failed ({exc})"

    if not hits:
        return f"No results for {query!r}."

    lines = [f"Search results for {query!r}:"]
    for i, hit in enumerate(hits, 1):
        title = hit.get("title", "(no title)")
        body = (hit.get("body") or "").strip().replace("\n", " ")
        url = hit.get("href", "")
        lines.append(f"{i}. {title}\n   {body}\n   source: {url}")
    return "\n".join(lines)


def read_file(path: str) -> str:
    """Read a UTF-8 text file, but only inside the workspace directory."""
    try:
        target = (WORKSPACE / path).resolve()
    except Exception as exc:
        return f"Error: bad path ({exc})"

    # Path traversal guard: '../../etc/passwd' resolves outside WORKSPACE.
    if not target.is_relative_to(WORKSPACE):
        return f"Error: refused — {path!r} is outside the workspace."
    if not target.is_file():
        return f"Error: no such file: {path}"

    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"Error: could not read file ({exc})"

    if len(text) > MAX_FILE_CHARS:
        text = text[:MAX_FILE_CHARS] + f"\n...[truncated at {MAX_FILE_CHARS} chars]"
    return f"Contents of {path}:\n{text}"


# --------------------------------------------------------------------------
# Schemas sent to the model (OpenAI-compatible format, which Ollama uses)
# --------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current local date and time. Use for any question about today's date, the time, or what day it is.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information. Use for recent events, news, prices, or any fact that may have changed since training.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Short search query, 1-6 words."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from the current project directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path, e.g. 'README.md'"}
                },
                "required": ["path"],
            },
        },
    },
]

# Name -> function. The loop uses this to dispatch; a name not in here is
# rejected rather than executed (models do occasionally invent tool names).
TOOL_REGISTRY = {
    "get_current_time": get_current_time,
    "web_search": web_search,
    "read_file": read_file,
}

# --- merged tool modules --------------------------------------------------
# Each module owns its own functions AND schemas, so adding a capability is
# one new file plus two lines here.
#
# TOOL BUDGET: 8 total. `calculator` was REMOVED in v3 because run_python
# does everything it did. Two overlapping math tools is exactly the ambiguity
# that makes a small model pick badly, and we already watched an 8B model
# fail to select web_search when there were only four options. Fewer, more
# distinct tools beat more, narrower ones.
from memory import MEMORY_TOOL_REGISTRY, MEMORY_TOOL_SCHEMAS      # noqa: E402
from research import RESEARCH_TOOL_REGISTRY, RESEARCH_TOOL_SCHEMAS  # noqa: E402
from sandbox import SANDBOX_TOOL_REGISTRY, SANDBOX_TOOL_SCHEMAS    # noqa: E402

TOOL_SCHEMAS = (
    TOOL_SCHEMAS
    + MEMORY_TOOL_SCHEMAS
    + RESEARCH_TOOL_SCHEMAS
    + SANDBOX_TOOL_SCHEMAS
)
TOOL_REGISTRY.update(MEMORY_TOOL_REGISTRY)
TOOL_REGISTRY.update(RESEARCH_TOOL_REGISTRY)
TOOL_REGISTRY.update(SANDBOX_TOOL_REGISTRY)
