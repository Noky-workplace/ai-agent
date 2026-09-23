"""
v5 — Python code execution, sandboxed with macOS Seatbelt.

WHAT THIS IS NOW: an OS-enforced boundary, not just process isolation.

Layers, outermost first:
  - Human approval. The code is shown and must be approved before it runs.
    This is still the primary control: it is the only layer that can catch
    "this code is legitimate but you don't want it run right now."
  - Seatbelt (sandbox-exec) with a deny-default profile in sandbox.sb:
    no network, no reads outside the interpreter and the scratch dir, no
    writes outside the scratch dir. This is kernel-enforced — an absolute
    path to ~/.ssh fails, where the temp cwd alone only stopped relative ones.
  - A separate process with a wall-clock timeout, a fresh temp cwd, and
    truncated output.

WHAT IT STILL DOES NOT DO:
  - limit memory or CPU beyond the timeout.
  - guarantee escape is impossible. Seatbelt profiles have been escaped in
    shipped tools: a zsh fork path in Codex CLI (fixed v0.106.0) and a
    glob/literal path confusion in Claude Code. Every `allow` rule is a place
    the profile can be wrong, which is why the approval prompt stays on.
  - protect anything on a non-macOS machine. Elsewhere this degrades to the
    old isolation-only behaviour and says so in the tool result.

Apple marks sandbox-exec deprecated and prints a warning on every call (we
filter it out below so it doesn't reach the model). It is still shipped and
is what Codex CLI and Claude Code use on macOS today.
"""

import os
import pathlib
import platform
import shutil
import subprocess
import sys
import tempfile

TIMEOUT_SECONDS = 15
MAX_OUTPUT_CHARS = 3000
MAX_CODE_CHARS = 8000

# Set False only if the agent is running inside a disposable container.
REQUIRE_APPROVAL = True

# Falls back to no sandbox (with a warning in the result) on other platforms
# or if the profile file is missing.
SANDBOX_EXEC = shutil.which("sandbox-exec") if platform.system() == "Darwin" else None
PROFILE = pathlib.Path(__file__).with_name("sandbox.sb")


def _command(script: pathlib.Path, tmpdir: str) -> tuple[list[str], bool]:
    """Build the subprocess argv, wrapped in sandbox-exec when available.

    realpath on every -D value matters: on macOS /var and /tmp are symlinks
    to /private/var and /private/tmp, and Seatbelt matches the resolved path.
    An unresolved TMP would silently match nothing and every write would fail.
    """
    base = [sys.executable, "-I", str(script)]
    if not SANDBOX_EXEC or not PROFILE.exists():
        return base, False
    return [
        SANDBOX_EXEC, "-f", str(PROFILE),
        f"-DTMP={os.path.realpath(tmpdir)}",
        f"-DPYBASE={os.path.realpath(sys.base_prefix)}",
        f"-DPYVENV={os.path.realpath(sys.prefix)}",
        "--", *base,
    ], True


def _truncate(text: str, limit: int, label: str) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[{label} truncated at {limit} chars]"


def _clean_stderr(text: str) -> str:
    """Drop the sandbox-exec deprecation banner.

    It appears on every single call. Left in, it would burn context on every
    run_python result and teach the model that something went wrong.
    """
    return "\n".join(
        line for line in text.splitlines()
        if "sandbox-exec is deprecated" not in line
    ).strip()


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
    """Execute Python in a sandboxed subprocess and return its output.

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

    # Fresh temp dir as cwd AND as the only writable path in the profile.
    # Deleted automatically when the block exits.
    with tempfile.TemporaryDirectory(prefix="agent-exec-") as tmpdir:
        script = pathlib.Path(tmpdir) / "snippet.py"
        script.write_text(code, encoding="utf-8")
        cmd, sandboxed = _command(script, tmpdir)

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=TIMEOUT_SECONDS, cwd=tmpdir,
            )
        except subprocess.TimeoutExpired:
            return (f"Error: execution exceeded {TIMEOUT_SECONDS}s and was killed. "
                    f"Likely an infinite loop or a very slow computation.")
        except Exception as exc:
            return f"Error: could not run code ({exc})."

    stdout = _truncate(proc.stdout.strip(), MAX_OUTPUT_CHARS, "stdout")
    stderr = _truncate(_clean_stderr(proc.stderr), MAX_OUTPUT_CHARS, "stderr")

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
    if not sandboxed:
        parts.append("(warning: ran WITHOUT sandbox-exec — no OS-level isolation)")

    return "\n\n".join(parts)


SANDBOX_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Run Python code and print output. Use for all arithmetic, "
                "data processing, and testing code. No network or file access "
                "outside its scratch directory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code. print() to see output.",
                    },
                },
                "required": ["code"],
            },
        },
    },
]

SANDBOX_TOOL_REGISTRY = {"run_python": run_python}
