#!/usr/bin/env python3
"""
v0 — Minimal local agent chat loop (Ollama, no frameworks).

From-scratch principles:
- Talk to Ollama over raw HTTP (no ollama/langchain packages) so you see
  exactly what an LLM API call is: an HTTP POST with a list of messages.
- One conversation = one growing list of {"role", "content"} dicts.
- Streaming: Ollama returns newline-delimited JSON chunks; we print each
  token as it arrives.

Prereqs (on your MacBook Air M4):
    brew install ollama          # or download from ollama.com
    ollama serve                 # usually auto-starts as a background app
    ollama pull qwen3:8b         # ~5.2 GB, fits 16 GB unified memory

Run:
    python3 chat.py
    python3 chat.py --model llama3.1:8b     # try other models

Commands inside the chat:
    /clear   wipe conversation history
    /system  show the current system prompt
    /exit    quit (Ctrl-C / Ctrl-D also work)
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"

# The system prompt is the first thing every agent designer should own.
# Keep it short for small models — they follow short instructions better.
SYSTEM_PROMPT = """You are a helpful personal assistant running fully locally.
Be concise. Think step by step for hard questions.
If you don't know something or it may have changed recently, say so honestly
instead of guessing — you do not have web access yet."""


def stream_chat(model: str, messages: list[dict]) -> str:
    """POST the conversation to Ollama and stream the reply to stdout.

    Returns the full assistant reply so the caller can append it to history.
    This function IS the core of every chat app you've ever used.
    """
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": True,
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    full_reply = []
    try:
        with urllib.request.urlopen(req) as resp:
            # Ollama streams newline-delimited JSON objects.
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    print(token, end="", flush=True)
                    full_reply.append(token)
                if chunk.get("done"):
                    break
    except urllib.error.URLError:
        print(
            "\n[error] Can't reach Ollama at localhost:11434.\n"
            "        Is it running? Try: ollama serve",
            file=sys.stderr,
        )
        return ""
    except KeyboardInterrupt:
        # Let the user cut off a rambling generation without killing the app.
        print("\n[interrupted]")

    print()  # final newline after the streamed reply
    return "".join(full_reply)


def main() -> None:
    parser = argparse.ArgumentParser(description="v0 local chat loop")
    parser.add_argument("--model", default="qwen3:8b",
                        help="Ollama model tag (default: qwen3:8b)")
    args = parser.parse_args()

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    print(f"Local agent v0 — model: {args.model}")
    print("Type /exit to quit, /clear to reset, /system to view the system prompt.\n")

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
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("[history cleared]")
            continue
        if user_input == "/system":
            print(f"[system prompt]\n{SYSTEM_PROMPT}")
            continue

        messages.append({"role": "user", "content": user_input})

        print("agent> ", end="", flush=True)
        reply = stream_chat(args.model, messages)

        if reply:
            # Appending the reply is what gives the model "memory" of the
            # conversation — there is no magic, just this list.
            messages.append({"role": "assistant", "content": reply})
        else:
            # Failed call: drop the user message so history stays consistent.
            messages.pop()


if __name__ == "__main__":
    main()
