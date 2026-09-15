"""
v5 — Safe file I/O.

Four tools: list_files, write_file, edit_file, delete_file. read_file lives
in tools.py but goes through the same `contain()` and `read_text()` below, so
the read side and the write side agree on what a path means.

THREAT MODEL
Both the path and the content come from the model, and the model can be
steered by anything it reads: a web page, a paper abstract, a note. So every
argument here is hostile input. Three layers:

  1. Containment.  Every path is joined to WORKSPACE and resolved with
     Path.resolve(), which follows symlinks (it is os.path.realpath under the
     hood). A string-only check like os.path.abspath never looks at the disk,
     so a symlink inside the workspace pointing outside passes it and open()
     then follows the link out. Three 2026 CVEs were exactly that bug:
     PraisonAI CVE-2026-55540, pgAdmin CVE-2026-7819, npm 'compressing'
     CVE-2026-40931.

  2. Denylist.  Secrets and agent-critical files are never readable or
     writable, whatever the user says. Compared case-folded and NFC-normalised
     because APFS is case-insensitive and normalisation-insensitive: '.ENV'
     and '.env' are the same file to the OS, so they must be the same file to
     the check. MEMORY.md is on the list because it is injected into the
     system prompt on every turn — a poisoned line there is persistent prompt
     injection. Only `remember` may touch it.

  3. Approval.  Reads are free. Every write, edit and delete shows the exact
     change (as a diff where possible) and asks. PLAN_MODE hard-disables
     mutation with no prompt and no override: it is a state, not a suggestion.

Also: deletes go to .trash/ (never unlink); writes are atomic (temp + rename);
every mutation is checkpointed into a shadow git repo first so /undo works;
every write is verified by re-reading. The Gemini CLI incident (Jul 2025)
destroyed a directory because the agent never ran a single check after
acting. Read-after-write is the cheapest fix for that class of failure.

O_NOFOLLOW on the final open() shrinks the check-then-use race: if the leaf
became a symlink between resolve() and open(), the open fails rather than
following it.
"""

import difflib
import fnmatch
import os
import pathlib
import subprocess
import time
import unicodedata

WORKSPACE = pathlib.Path.cwd().resolve()

MAX_WRITE_CHARS = 20_000
MAX_READ_BYTES = 2_000_000
MAX_LIST_ENTRIES = 60
PREVIEW_LINES = 30
TRASH_DIR = ".trash"
SHADOW_GIT = ".agent-git"

# Set by agent.py --plan. When True, write/edit/delete refuse without asking.
PLAN_MODE = False

# Never readable or writable by the model. Each pattern is matched against
# every component of the relative path, case-folded and NFC-normalised.
PROTECTED = (
    ".env", ".env.*",
    "*.pem", "*.key", "*.p12", "id_rsa*", "id_ed25519*", ".ssh",
    ".git", ".agent-git", ".venv", "__pycache__", ".trash",
    "MEMORY.md",
)
HIDDEN_FROM_LISTING = (".git", ".agent-git", ".venv", "__pycache__",
                       ".trash", ".DS_Store")

_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)

DENIED = ("DENIED: the user declined this change. Do not retry the same "
          "change. Explain what you wanted to do, or propose something different.")
PLAN_REFUSED = ("REFUSED: plan mode is on, so no file changes are allowed "
                "this session. Describe the change you would make instead.")


class Refused(Exception):
    """Raised when a path fails containment or the denylist."""


# --------------------------------------------------------------------------
# Containment
# --------------------------------------------------------------------------

def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", s).casefold()


def _is_protected(rel: pathlib.Path) -> bool:
    pats = [_norm(p) for p in PROTECTED]
    return any(
        fnmatch.fnmatchcase(_norm(part), pat)
        for part in rel.parts
        for pat in pats
    )


def contain(path: str) -> pathlib.Path:
    """Map a model-supplied path to a real path inside WORKSPACE, or raise.

    Resolves symlinks BEFORE the prefix check. Then applies the denylist to
    the resolved relative path, so 'notes/../.env' and a symlink named
    'safe.txt' pointing at .env are both caught.
    """
    raw = (path or "").strip()
    if not raw or "\x00" in raw:
        raise Refused("bad path.")
    try:
        target = (WORKSPACE / raw).resolve()
    except Exception:
        raise Refused("bad path.")
    if not target.is_relative_to(WORKSPACE):
        raise Refused(f"{raw!r} is outside the workspace.")
    rel = target.relative_to(WORKSPACE)
    if rel.parts and _is_protected(rel):
        raise Refused(f"{raw!r} is protected.")
    return target


def _rel(target: pathlib.Path) -> str:
    return str(target.relative_to(WORKSPACE))


# --------------------------------------------------------------------------
# Low-level I/O (O_NOFOLLOW, atomic write, binary detection)
# --------------------------------------------------------------------------

def read_text(target: pathlib.Path) -> str:
    """Read a text file via a NOFOLLOW handle. Raises Refused on binary."""
    fd = os.open(target, os.O_RDONLY | _NOFOLLOW)
    with os.fdopen(fd, "rb") as fh:
        data = fh.read(MAX_READ_BYTES)
    if b"\x00" in data[:8192]:
        raise Refused("that is a binary file.")
    return data.decode("utf-8", errors="replace")


def _atomic_write(target: pathlib.Path, text: str) -> None:
    """Write to a sibling temp file, then rename over the target.

    os.replace is atomic on the same filesystem, and it replaces a symlink
    rather than following it, so a link swapped in after contain() cannot
    redirect the write.
    """
    tmp = target.with_name(f".{target.name}.agent-tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW, 0o644)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, target)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


# --------------------------------------------------------------------------
# Approval gate and shadow-git checkpoints
# --------------------------------------------------------------------------

def _ask(title: str, lines: list[str]) -> bool:
    """Show what will happen and ask. Deny on anything but an explicit yes."""
    print("\n" + "=" * 62)
    print(f"  {title}")
    print("=" * 62)
    for line in lines[:PREVIEW_LINES]:
        print(f"  | {line}")
    if len(lines) > PREVIEW_LINES:
        print(f"  | ... ({len(lines) - PREVIEW_LINES} more lines)")
    print("=" * 62)
    try:
        answer = input("  Allow? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in ("y", "yes")


def _git() -> list[str]:
    return ["git", f"--git-dir={WORKSPACE / SHADOW_GIT}", f"--work-tree={WORKSPACE}"]


def _checkpoint(reason: str) -> None:
    """Commit the workspace to a shadow repo (.agent-git) before mutating.

    Separate from the user's real .git so agent noise never lands in their
    history; git refuses to track a nested .git directory, so the real repo
    is skipped automatically. Best-effort: a failure is printed, not fatal,
    because refusing to edit when git hiccups would be worse than editing
    without a checkpoint.
    """
    git = _git()
    try:
        if not (WORKSPACE / SHADOW_GIT).exists():
            subprocess.run(git + ["init", "-q"], check=True, capture_output=True)
            exclude = WORKSPACE / SHADOW_GIT / "info" / "exclude"
            exclude.parent.mkdir(exist_ok=True)
            exclude.write_text("\n".join(HIDDEN_FROM_LISTING + (".env", ".env.*")) + "\n")
        subprocess.run(git + ["add", "-A"], check=True, capture_output=True, timeout=20)
        subprocess.run(
            git + ["-c", "user.name=agent", "-c", "user.email=agent@local",
                   "commit", "-q", "--allow-empty", "-m", f"checkpoint: {reason}"],
            check=True, capture_output=True, timeout=20,
        )
    except Exception as exc:
        print(f"  [checkpoint failed: {exc}]")


def undo_last() -> str:
    """Restore tracked files to the most recent checkpoint. Used by /undo.

    Restores every file the shadow repo knows about, including any you
    edited by hand since the checkpoint. Files the agent CREATED after the
    checkpoint are untracked and are left in place — delete those manually.
    """
    git = _git()
    if not (WORKSPACE / SHADOW_GIT).exists():
        return "Nothing to undo: no checkpoints yet."
    try:
        subject = subprocess.run(git + ["log", "-1", "--format=%s"],
                                 capture_output=True, text=True, check=True).stdout.strip()
        subprocess.run(git + ["checkout", "-q", "HEAD", "--", "."],
                       check=True, capture_output=True)
        # Step HEAD back one so a second /undo goes one checkpoint further.
        subprocess.run(git + ["reset", "-q", "--soft", "HEAD~1"], capture_output=True)
    except subprocess.CalledProcessError as exc:
        return f"Undo failed: {exc.stderr.decode(errors='replace').strip()}"
    return f"Restored workspace to '{subject}'. New files created since then were left in place."


def _diff(old: str, new: str, rel: str) -> list[str]:
    return list(difflib.unified_diff(
        old.splitlines(), new.splitlines(), f"a/{rel}", f"b/{rel}", lineterm="", n=2,
    ))


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------

def list_files(path: str = ".") -> str:
    """List a directory inside the workspace. Read-only, no approval."""
    try:
        target = contain(path)
    except Refused as exc:
        return f"Error: refused — {exc}"
    if not target.is_dir():
        return f"Error: not a directory: {path}"

    entries = []
    try:
        children = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except Exception as exc:
        return f"Error: could not list ({exc})"
    for child in children:
        if child.name in HIDDEN_FROM_LISTING or _is_protected(child.relative_to(WORKSPACE)):
            continue
        rel = _rel(child)
        if child.is_symlink():
            entries.append(f"{rel}  (symlink)")
        elif child.is_dir():
            entries.append(f"{rel}/")
        else:
            entries.append(f"{rel}  ({child.stat().st_size} bytes)")

    if not entries:
        return f"{_rel(target) or '.'} is empty."
    shown = entries[:MAX_LIST_ENTRIES]
    tail = f"\n...({len(entries) - MAX_LIST_ENTRIES} more)" if len(entries) > MAX_LIST_ENTRIES else ""
    return "\n".join(shown) + tail


def write_file(path: str, content: str) -> str:
    """Create or fully overwrite a text file. Asks first; verifies after."""
    if PLAN_MODE:
        return PLAN_REFUSED
    try:
        target = contain(path)
    except Refused as exc:
        return f"Error: refused — {exc}"
    content = content if isinstance(content, str) else str(content)
    if len(content) > MAX_WRITE_CHARS:
        return f"Error: content too long (limit {MAX_WRITE_CHARS} chars)."
    if target == WORKSPACE or target.is_dir():
        return f"Error: {path!r} is a directory."

    rel = _rel(target)
    if target.exists():
        try:
            old = read_text(target)
        except Refused as exc:
            return f"Error: refused — {exc}"
        except Exception as exc:
            return f"Error: could not read existing file ({exc})"
        title, preview = f"The agent wants to OVERWRITE {rel}", _diff(old, content, rel)
        if not preview:
            return f"No change: {rel} already has that content."
    else:
        title, preview = f"The agent wants to CREATE {rel}", content.splitlines()

    if not _ask(title, preview):
        return DENIED

    _checkpoint(f"before write {rel}")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(target, content)
        back = read_text(target)
    except Exception as exc:
        return f"Error: write failed ({exc})"
    if back != content:
        return "Error: verification failed — the file on disk does not match what was written."
    return f"Wrote {len(content)} chars to {rel} (verified)."


def edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """Exact-string replace. Refuses ambiguous edits rather than guessing."""
    if PLAN_MODE:
        return PLAN_REFUSED
    try:
        target = contain(path)
    except Refused as exc:
        return f"Error: refused — {exc}"
    if not target.is_file():
        return f"Error: no such file: {path}"
    if not old_string:
        return "Error: old_string is empty."
    new_string = new_string if isinstance(new_string, str) else str(new_string)

    rel = _rel(target)
    try:
        old = read_text(target)
    except Refused as exc:
        return f"Error: refused — {exc}"
    except Exception as exc:
        return f"Error: could not read file ({exc})"

    count = old.count(old_string)
    if count == 0:
        return (f"Error: old_string not found in {rel}. read_file it and copy the "
                f"text exactly — whitespace and indentation must match.")
    if count > 1 and not replace_all:
        return (f"Error: old_string appears {count} times in {rel}. Include more "
                f"surrounding lines to make it unique, or set replace_all=true.")

    new = old.replace(old_string, new_string) if replace_all else old.replace(old_string, new_string, 1)
    if new == old:
        return f"No change: replacement is identical to the original in {rel}."

    n = count if replace_all else 1
    if not _ask(f"The agent wants to EDIT {rel} ({n} replacement{'s' if n > 1 else ''})",
                _diff(old, new, rel)):
        return DENIED

    _checkpoint(f"before edit {rel}")
    try:
        _atomic_write(target, new)
        back = read_text(target)
    except Exception as exc:
        return f"Error: write failed ({exc})"
    if back != new:
        return "Error: verification failed — the file on disk does not match the intended edit."
    return f"Replaced {n} of {count} occurrence(s) in {rel} (verified)."


def delete_file(path: str) -> str:
    """Move a file to .trash/ inside the workspace. Never unlinks."""
    if PLAN_MODE:
        return PLAN_REFUSED
    try:
        target = contain(path)
    except Refused as exc:
        return f"Error: refused — {exc}"
    if not target.is_file():
        return f"Error: not a file (directories cannot be deleted): {path}"

    rel = _rel(target)
    size = target.stat().st_size
    if not _ask(f"The agent wants to DELETE {rel}", [f"{size} bytes — will be moved to {TRASH_DIR}/"]):
        return DENIED

    _checkpoint(f"before delete {rel}")
    trash = WORKSPACE / TRASH_DIR
    try:
        trash.mkdir(exist_ok=True)
        dest = trash / f"{time.strftime('%Y%m%d-%H%M%S')}-{target.name}"
        target.rename(dest)
    except Exception as exc:
        return f"Error: delete failed ({exc})"
    if target.exists():
        return "Error: verification failed — the file still exists."
    return f"Moved {rel} to {TRASH_DIR}/{dest.name}. It can be restored from there."


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------

FILES_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and folders in a directory of the project.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative directory, e.g. '.' or 'notes'."}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create a new text file, or fully replace an existing one. "
                "For changing part of a file use edit_file instead. The user "
                "must approve."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path, e.g. 'notes/plan.md'."},
                    "content": {"type": "string", "description": "Full file content."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Replace an exact string in a text file with new text. Call "
                "read_file first and copy old_string exactly, including "
                "whitespace. old_string must be unique unless replace_all is "
                "true. The user must approve."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path."},
                    "old_string": {"type": "string", "description": "Exact text to find."},
                    "new_string": {"type": "string", "description": "Replacement text."},
                    "replace_all": {"type": "boolean", "description": "Replace every occurrence. Default false."},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "Move a file to the project trash folder. The user must approve.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path."},
                },
                "required": ["path"],
            },
        },
    },
]

FILES_TOOL_REGISTRY = {
    "list_files": list_files,
    "write_file": write_file,
    "edit_file": edit_file,
    "delete_file": delete_file,
}
