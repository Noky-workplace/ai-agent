#!/usr/bin/env python3
"""
v3 — Local agent with tool calling + file-based memory (hand-written ReAct loop, no frameworks).

THE LOOP (this is the whole idea of an agent):

    1. Send [conversation + tool schemas] to the model.
    2. Model replies with EITHER a final answer OR one/more tool calls.
    3. If tool calls: run them, append results as role="tool" messages, goto 1.
    4. If final answer: print it, wait for the user.
    5. Never loop more than MAX_ITERATIONS times.

Step 5 matters. A confused small model will happily search the same thing
forever. Every production agent has this cap.

Run:
    python3 agent.py
    python3 agent.py --model qwen3:14b
    python3 agent.py --quiet          # hide tool-call traces

Requires: pip install ddgs
"""

import argparse
import json
import sys
import re
import time
import urllib.error
import urllib.request

from memory import load_memory
from tools import TOOL_REGISTRY, TOOL_SCHEMAS

OLLAMA_URL = "http://localhost:11434/api/chat"
MAX_ITERATIONS = 6

# Ollama picks a context length from available VRAM, which on a 16GB Mac came
# out at only 4096 tokens. That is a real problem: the system prompt plus 8
# tool schemas is already ~1200-1400 tokens of FIXED overhead on every single
# request, so a third of the window is gone before the user types anything.
# One paper search returning 5 abstracts can add another ~700. Overflow drops
# the OLDEST messages first -- which is the system prompt, i.e. exactly the
# rules like "never do arithmetic in your head" that we depend on.
#
# Qwen3 supports far more than 4096 natively. 16384 is a reasonable default
# on 16GB with an 8B model. Raise it if you have headroom, lower it if the
# model spills into system RAM and generation crawls.
DEFAULT_NUM_CTX = 16384

BASE_PROMPT = """You are a helpful personal assistant running fully locally on the user's Mac.

You have tools. Use them instead of guessing:
- Today's date or time -> get_current_time
- Recent events, news, prices, or facts that may have changed -> web_search
- Papers, studies, research literature -> search_papers
- ANY arithmetic, data processing, or code testing -> run_python
  (there is no calculator tool; write Python and print the result)
- Reading a file, including a note -> read_file
- A short lasting fact about the USER (name, school, preferences) -> remember
- Longer subject content worth keeping (research, lecture notes) -> write_note
  (never use write_note to store facts about the user — that is remember's job)
- Finding something saved earlier -> search_notes, then read_file the result

Rules:
- Your training data is out of date. If a question is about the current state
  of the world, search before answering. Do not answer from memory.
- Call one tool at a time, then look at the result before deciding what next.
- search_notes is literal keyword matching, not semantic. If a search returns
  nothing, try a different word before concluding the note does not exist.
- Only call remember when the fact is durable and about the user. Do not save
  passing conversational details.
- If remember reports the fact overlaps an existing memory, decide: if the
  new fact is more specific, call remember again with 'replaces' set to the
  existing text. Do not create a second near-identical memory.
- When you have enough information, give a concise final answer and cite the
  source URLs you used.
- If a tool returns an error, tell the user plainly. Do not invent the answer.
- Never do arithmetic in your head. Write it as Python and run it.
- If the user denies a run_python request, respect it. Do not resubmit the
  same code. Explain in text or propose something different."""


def build_system_prompt() -> str:
    """Inject MEMORY.md into the system prompt.

    This is the whole 'long-term memory' mechanism: read a file at startup,
    paste it into the prompt. No embeddings, no retrieval step, no database.
    """
    memory = load_memory()
    if not memory.strip():
        return BASE_PROMPT
    return (
        f"{BASE_PROMPT}\n\n"
        "--- Loaded from MEMORY.md ---\n"
        "The facts below were saved from earlier conversations. They live in a\n"
        "file called MEMORY.md in the project folder — they are NOT part of your\n"
        "training data, and they CAN be changed. Use the remember tool to add or\n"
        "update them. Never tell the user these facts are fixed or unmodifiable.\n"
        "You already know everything below, so answer questions about the user\n"
        "directly from it without searching notes first.\n\n"
        f"{memory}"
    )


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English.

    Crude on purpose -- the real tokenizer lives in the model, and we only
    need order-of-magnitude to spot when overhead is eating the window.
    """
    return len(text) // 4


def context_report(system_prompt: str, messages: list[dict], num_ctx: int) -> str:
    """Show what is actually consuming the context window."""
    sys_t = estimate_tokens(system_prompt)
    schema_t = estimate_tokens(json.dumps(TOOL_SCHEMAS))
    convo_t = sum(estimate_tokens(str(m.get("content") or "")) for m in messages[1:])
    total = sys_t + schema_t + convo_t
    pct = (total / num_ctx * 100) if num_ctx else 0
    return (
        f"[context ~{total}/{num_ctx} tokens ({pct:.0f}%)]\n"
        f"  system prompt : ~{sys_t}\n"
        f"  {len(TOOL_SCHEMAS)} tool schemas: ~{schema_t}  (fixed cost every request)\n"
        f"  conversation  : ~{convo_t}"
    )


def call_ollama(model: str, messages: list[dict], num_ctx: int = DEFAULT_NUM_CTX,
                think: bool = False) -> dict:
    """One non-streaming POST to Ollama. Returns the assistant message dict.

    We use stream=False here (unlike v0) because tool calls arrive as a
    structured field, and reassembling them from stream chunks is fiddly.
    Correctness first; you can add streaming back for the final answer later.
    """
    body = {
        "model": model,
        "messages": messages,
        "tools": TOOL_SCHEMAS,
        "stream": False,
        "options": {"num_ctx": num_ctx},
        # Qwen3 is a HYBRID REASONING model: by default it emits a long
        # internal <think> block before every answer. Those tokens are
        # generated at the same speed as visible ones, so thinking can easily
        # triple or quadruple the wall-clock time of a turn -- and in a ReAct
        # loop you pay it on EVERY iteration, not once per user message.
        #
        # For tool-calling work, thinking is usually poor value: the decision
        # "which tool do I call" rarely needs paragraphs of deliberation.
        # Turn it back on (--think) for genuinely hard reasoning questions.
        "think": think,
    }
    payload = json.dumps(body).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body.get("message", {})


def execute_tool(name: str, args: dict) -> str:
    """Dispatch a model-requested tool call. Never trusts the model blindly."""
    func = TOOL_REGISTRY.get(name)
    if func is None:
        # Models sometimes hallucinate tool names. Tell it, don't crash.
        return f"Error: no tool named {name!r}. Available: {', '.join(TOOL_REGISTRY)}"
    if not isinstance(args, dict):
        return f"Error: arguments for {name} must be an object."
    try:
        return str(func(**args))
    except TypeError as exc:
        return f"Error: wrong arguments for {name} ({exc})"
    except Exception as exc:
        return f"Error: {name} failed ({exc})"


def run_agent(model: str, messages: list[dict], verbose: bool = True,
              num_ctx: int = DEFAULT_NUM_CTX, think: bool = False,
              show_time: bool = False) -> None:
    """The ReAct loop. Mutates `messages` in place so history persists."""
    for step in range(1, MAX_ITERATIONS + 1):
        try:
            t0 = time.time()
            reply = call_ollama(model, messages, num_ctx, think)
            elapsed = time.time() - t0
            if show_time:
                out_chars = len(str(reply.get("content") or ""))
                rate = (out_chars / 4) / elapsed if elapsed else 0
                print(f"  [ollama call: {elapsed:.1f}s, ~{rate:.0f} tok/s]")
        except urllib.error.URLError:
            print("[error] Can't reach Ollama at localhost:11434. Is it running?",
                  file=sys.stderr)
            return
        except Exception as exc:
            print(f"[error] Ollama call failed: {exc}", file=sys.stderr)
            return

        messages.append(reply)
        tool_calls = reply.get("tool_calls") or []

        if not tool_calls:
            # No tools requested -> this is the final answer.
            content = (reply.get("content") or "").strip()
            # Some builds return the reasoning inline; never show it.
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
            print(f"agent> {content}\n" if content else "agent> [empty reply]\n")
            return

        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            # Some models return arguments as a JSON string instead of an object.
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            if verbose:
                pretty = ", ".join(f"{k}={v!r}" for k, v in args.items())
                print(f"  [{step}] -> {name}({pretty})")

            result = execute_tool(name, args)

            if verbose:
                preview = result.replace("\n", " ")[:110]
                print(f"      <- {preview}{'...' if len(result) > 110 else ''}")

            # role="tool" is how the model learns what happened.
            messages.append({"role": "tool", "name": name, "content": result})

    print(f"agent> [stopped after {MAX_ITERATIONS} steps without a final answer]\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="v3 local agent: tools, file memory, papers, code execution")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--quiet", action="store_true", help="hide tool traces")
    parser.add_argument("--ctx", type=int, default=DEFAULT_NUM_CTX,
                        help=f"context window in tokens (default {DEFAULT_NUM_CTX}). "
                             f"Use --ctx 4096 to reproduce Ollama's VRAM default.")
    parser.add_argument("--think", action="store_true",
                        help="enable Qwen3 reasoning mode (much slower)")
    parser.add_argument("--time", action="store_true",
                        help="print seconds per Ollama call")
    args = parser.parse_args()

    system_prompt = build_system_prompt()
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    mem_status = "loaded" if load_memory().strip() else "empty"
    print(f"Local agent v3 — model: {args.model}  ctx: {args.ctx}  (MEMORY.md: {mem_status})")
    overhead = estimate_tokens(system_prompt) + estimate_tokens(json.dumps(TOOL_SCHEMAS))
    print(f"Fixed overhead: ~{overhead} tokens ({overhead/args.ctx*100:.0f}% of context)")
    print(f"Tools: {', '.join(TOOL_REGISTRY)}")
    print("Commands: /clear  /history  /memory  /context  /exit\n")

    while True:
        try:
            user_input = input("you> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nbye")
            break

        if not user_input:
            continue
        if user_input == "/exit":
            print("bye")
            break
        if user_input == "/clear":
            system_prompt = build_system_prompt()  # re-read MEMORY.md
            messages = [{"role": "system", "content": system_prompt}]
            print("[history cleared, memory reloaded]\n")
            continue
        if user_input == "/context":
            print(context_report(system_prompt, messages, args.ctx) + "\n")
            continue
        if user_input == "/memory":
            mem = load_memory()
            print(f"[MEMORY.md]\n{mem}\n" if mem.strip() else "[MEMORY.md is empty]\n")
            continue
        if user_input == "/history":
            print(f"[{len(messages)} messages in context]\n")
            continue

        messages.append({"role": "user", "content": user_input})
        run_agent(args.model, messages, verbose=not args.quiet,
                  num_ctx=args.ctx, think=args.think, show_time=args.time)


if __name__ == "__main__":
    main()