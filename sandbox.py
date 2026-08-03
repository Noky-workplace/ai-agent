"""
v3 — Python code execution.

READ THIS BEFORE USING: this is ISOLATION, NOT A SECURITY SANDBOX.

What it actually does:
  - runs code in a separate process (a crash can't take down the agent)
  - kills it after a timeout (infinite loops can't hang you)
  - runs it in a fresh temp directory (relative paths can't touch your project)
  - truncates output (a runaway print loop can't flood your context)
  - asks YOU to approve the code before it runs

What it does NOT do:
  - block filesystem access outside the temp dir. `open('/Users/you/.ssh/id_ed25519')`
    works. The temp cwd only stops *relative* paths.
  - block network access. The code can make HTTP requests.
  - limit memory or CPU beyond wall-clock time.

Real isolation needs an OS-level boundary — Docker with --network=none and a
read-only mount, or a VM, or gVisor. Those are the right answer if this agent
ever runs code you did not read first.

The human approval prompt is therefore not a nicety, it is the actual
security control. Keep it on unless you are running in a throwaway container.
This is also a preview of v4: the general principle is that tools with real
side effects need a confirmation step, because the model can be talked into
things by content it reads.
"""

import pathlib
import subprocess
import sys
import tempfile

TIMEOUT_SECONDS = 15
MAX_OUTPUT_CHARS = 3000
MAX_CODE_CHARS = 8000

# Set False only if the agent is running inside a disposable container.
REQUIRE_APPROVAL = True


def _truncate(text: str, limit: int, label: str) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[{label} truncated at {limit} chars]"


def _ask_approval(code: str) -> bool:
    """Show the code and ask the human. Deny on anything but an explicit yes."""
    print("\n" + "=" * 62)
    print("  The agent wants to run this Python code:")
    print("=" * 62)
    for line in code.splitlines():
        print(f"  | {line}")
    print("=" * 62)
    try:
        answer = input("  Run it? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in ("y", "yes")


def run_python(code: str) -> str:
    """Execute Python in a subprocess and return its output.

    Use for calculations, data manipulation, algorithm testing — anything
    where running the code beats reasoning about what it would print.
    """
    code = (code or "").strip()
    if not code:
        return "Error: no code provided."
    if len(code) > MAX_CODE_CHARS:
        return f"Error: code too long (limit {MAX_CODE_CHARS} chars)."

    if REQUIRE_APPROVAL and not _ask_approval(code):
        return ("DENIED: the user declined to run this code. "
                "Do not retry the same code. Explain your reasoning in text "
                "instead, or propose different code and ask again.")

    # Fresh temp dir as cwd: relative file operations land here, not in the
    # user's project. Deleted automatically when the block exits.
    with tempfile.TemporaryDirectory(prefix="agent-exec-") as tmpdir:
        script = pathlib.Path(tmpdir) / "snippet.py"
        script.write_text(code, encoding="utf-8")

        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],  # -I = isolated mode
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                cwd=tmpdir,
            )
        except subprocess.TimeoutExpired:
            return (f"Error: execution exceeded {TIMEOUT_SECONDS}s and was killed. "
                    f"Likely an infinite loop or a very slow computation.")
        except Exception as exc:
            return f"Error: could not run code ({exc})."

    stdout = _truncate(proc.stdout.strip(), MAX_OUTPUT_CHARS, "stdout")
    stderr = _truncate(proc.stderr.strip(), MAX_OUTPUT_CHARS, "stderr")

    parts = []
    if stdout:
        parts.append(f"stdout:\n{stdout}")
    if stderr:
        # Non-zero exit with a traceback is useful signal, not a failure of
        # the tool — hand it back so the model can fix its own code.
        parts.append(f"stderr:\n{stderr}")
    if not parts:
        parts.append("(no output — did you forget to print?)")
    if proc.returncode != 0:
        parts.append(f"exit code: {proc.returncode}")

    return "\n\n".join(parts)


SANDBOX_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Run Python code and return its output. Use for ALL arithmetic "
                "and calculations, data processing, algorithm testing, and "
                "verifying code works. You must print() anything you want to "
                "see. The code runs in an isolated temp directory with a "
                f"{TIMEOUT_SECONDS}s limit, and the user must approve it first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code. Use print() to produce output.",
                    },
                },
                "required": ["code"],
            },
        },
    },
]

SANDBOX_TOOL_REGISTRY = {"run_python": run_python}
