"""
v5 — Python code execution, sandboxed with macOS Seatbelt.

WHAT THIS IS NOW: an OS-enforced boundary, not just process isolation.

Layers, outermost first:
  - Human approval. The code is shown and must be approved before it runs.
    This is still the primary control: it is the only layer that can catch
    "this code is legitimate but you don't want it run right now."
  - Seatbelt (sandbox-exec) with a deny-default profile in sandbox.sb:
    no network, no reads outside the interpreter and sandbox_io/, no
    writes outside sandbox_io/. This is kernel-enforced — an absolute
    path to ~/.ssh fails, and so does a write to agent.py or MEMORY.md.
  - A separate process with a wall-clock timeout and truncated output.

SANDBOX_IO
sandbox_io/ in the workspace is the sandbox's cwd and its only writable
path, and it persists across calls. Because it is inside the workspace the
file tools see it too: write_file stages data in, run_python computes,
read_file reads results out. extract_archive unpacks into it.

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
import tarfile
import tempfile
import zipfile

import files

TIMEOUT_SECONDS = 15
MAX_OUTPUT_CHARS = 3000
MAX_CODE_CHARS = 8000

# Set False only if the agent is running inside a disposable container.
REQUIRE_APPROVAL = True

# Falls back to no sandbox (with a warning in the result) on other platforms
# or if the profile file is missing.
SANDBOX_EXEC = shutil.which("sandbox-exec") if platform.system() == "Darwin" else None
PROFILE = pathlib.Path(__file__).with_name("sandbox.sb")

IO_DIR_NAME = "sandbox_io"


def io_dir() -> pathlib.Path:
    """sandbox_io/ inside the workspace, created on demand."""
    path = files.WORKSPACE / IO_DIR_NAME
    path.mkdir(exist_ok=True)
    return path


def _command(script: pathlib.Path, tmpdir: str, iodir: pathlib.Path) -> tuple[list[str], bool]:
    """Build the subprocess argv, wrapped in sandbox-exec when available.

    realpath on every -D value matters: on macOS /var and /tmp are symlinks
    to /private/var and /private/tmp, and Seatbelt matches the resolved path.
    An unresolved IO would silently match nothing and every write would fail.
    """
    base = [sys.executable, "-I", str(script)]
    if not SANDBOX_EXEC or not PROFILE.exists():
        return base, False
    return [
        SANDBOX_EXEC, "-f", str(PROFILE),
        f"-DTMP={os.path.realpath(tmpdir)}",
        f"-DIO={os.path.realpath(iodir)}",
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
    print("\n" + "=" * 62, file=sys.__stdout__)
    print("  The agent wants to run this Python code:", file=sys.__stdout__)
    print("=" * 62, file=sys.__stdout__)
    for line in code.splitlines():
        print(f"  | {line}", file=sys.__stdout__)
    print("=" * 62, file=sys.__stdout__)
    try:
        # input()'s own prompt would go to the redirected sys.stdout.
        print("  Run it? [y/N] ", end="", flush=True, file=sys.__stdout__)
        answer = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        print(file=sys.__stdout__)
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

    # The script lives in a per-call temp dir the sandbox can only read.
    # cwd is sandbox_io/, which persists, so step 2 sees what step 1 wrote.
    iodir = io_dir()
    with tempfile.TemporaryDirectory(prefix="agent-exec-") as tmpdir:
        script = pathlib.Path(tmpdir) / "snippet.py"
        script.write_text(code, encoding="utf-8")
        cmd, sandboxed = _command(script, tmpdir, iodir)

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=TIMEOUT_SECONDS, cwd=iodir,
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


# --------------------------------------------------------------------------
# extract_archive
# --------------------------------------------------------------------------
#
# The member checks below are defense in depth for an in-process extract.
# Zip-slip through run_python is already unexploitable: the kernel refuses
# any write outside sandbox_io/, whatever the archive says.

MAX_ARCHIVE_BYTES = 200_000_000    # total uncompressed
MAX_ARCHIVE_MEMBERS = 2000
MAX_COMPRESSION_RATIO = 100        # per zip member; whole stream for tar
RATIO_FLOOR_BYTES = 1_000_000      # tiny text compresses >100x legitimately
MAX_LANDED_LISTING = 60

_TAR_SUFFIXES = (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".tbz2", ".txz", ".tar")


def _archive_kind(name: str) -> tuple[str, str] | None:
    """('zip'|'tar', stem) from the filename, or None if unsupported."""
    low = name.lower()
    if low.endswith(".zip"):
        return "zip", name[:-4]
    for suffix in _TAR_SUFFIXES:
        if low.endswith(suffix):
            return "tar", name[:-len(suffix)]
    return None


def _unsafe_name(name: str) -> bool:
    n = name.replace("\\", "/")
    return n.startswith("/") or n[1:2] == ":" or ".." in n.split("/")


def _check_zip(zf: zipfile.ZipFile) -> tuple[list, int]:
    infos = zf.infolist()
    if len(infos) > MAX_ARCHIVE_MEMBERS:
        raise files.Refused(f"too many entries ({len(infos)} > {MAX_ARCHIVE_MEMBERS}).")
    total = 0
    for info in infos:
        if _unsafe_name(info.filename):
            raise files.Refused(f"unsafe entry path {info.filename!r}.")
        if (info.external_attr >> 16) & 0o170000 == 0o120000:
            raise files.Refused(f"symlink entry {info.filename!r}.")
        if info.file_size > RATIO_FLOOR_BYTES and (not info.compress_size
                               or info.file_size / info.compress_size > MAX_COMPRESSION_RATIO):
            raise files.Refused(f"entry {info.filename!r} exceeds the "
                                f"{MAX_COMPRESSION_RATIO}x compression ratio limit.")
        total += info.file_size
    return infos, total


def _check_tar(tf: tarfile.TarFile, archive_size: int) -> tuple[list, int]:
    members, total = [], 0
    for member in tf:      # iterate, so a huge member count stops early
        members.append(member)
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise files.Refused(f"too many entries (> {MAX_ARCHIVE_MEMBERS}).")
        if _unsafe_name(member.name):
            raise files.Refused(f"unsafe entry path {member.name!r}.")
        if not (member.isfile() or member.isdir()):
            raise files.Refused(f"link or special entry {member.name!r}.")
        total += member.size
        if total > MAX_ARCHIVE_BYTES:
            break
    if total > RATIO_FLOOR_BYTES and total / max(archive_size, 1) > MAX_COMPRESSION_RATIO:
        raise files.Refused(f"exceeds the {MAX_COMPRESSION_RATIO}x compression ratio limit.")
    return members, total


def extract_archive(path: str) -> str:
    """Unpack a zip or tar archive from the workspace into sandbox_io/<stem>/."""
    if files.PLAN_MODE:
        return files.PLAN_REFUSED
    try:
        src = files.contain(path)
    except files.Refused as exc:
        return f"Error: refused — {exc}"
    if not src.is_file():
        return f"Error: no such file: {path}"
    kind = _archive_kind(src.name)
    if kind is None:
        return "Error: unsupported archive type (zip, tar, tar.gz, tar.bz2, tar.xz)."
    fmt, stem = kind

    iodir = io_dir()
    if not iodir.resolve().is_relative_to(files.WORKSPACE):
        return f"Error: refused — {IO_DIR_NAME}/ resolves outside the workspace."
    dest = iodir / stem
    dest_rel = f"{IO_DIR_NAME}/{stem}/"
    if os.path.lexists(dest):
        return (f"Error: {dest_rel} already exists. Use the existing files, "
                f"or delete them first.")

    try:
        archive = zipfile.ZipFile(src) if fmt == "zip" else tarfile.open(src)
    except (zipfile.BadZipFile, tarfile.TarError, OSError) as exc:
        return f"Error: could not open archive ({exc})"

    with archive:
        try:
            if fmt == "zip":
                members, total = _check_zip(archive)
                names = [m.filename for m in members]
            else:
                members, total = _check_tar(archive, src.stat().st_size)
                names = [m.name for m in members]
            if total > MAX_ARCHIVE_BYTES:
                raise files.Refused(f"uncompressed size exceeds {MAX_ARCHIVE_BYTES:,} bytes.")
        except files.Refused as exc:
            return f"Error: refused — {exc}"
        except (zipfile.BadZipFile, tarfile.TarError, OSError) as exc:
            return f"Error: could not read archive ({exc})"

        rel = files._rel(src)
        summary = f"{len(members)} entries, {total:,} bytes uncompressed"
        if not files._ask(f"The agent wants to EXTRACT {rel} into {dest_rel}",
                          [summary, *names]):
            return files.DENIED

        try:
            dest.mkdir()    # fails if a symlink appeared at dest meanwhile
            if fmt == "zip":
                archive.extractall(dest, members=members)
            else:
                archive.extractall(dest, members=members, filter="data")
        except Exception as exc:
            return f"Error: extraction failed partway ({exc}). Partial output is in {dest_rel}."

    landed = sorted(str(p.relative_to(iodir)) for p in dest.rglob("*") if p.is_file())
    shown = landed[:MAX_LANDED_LISTING]
    more = len(landed) - len(shown)
    listing = "\n".join(f"  {name}" for name in shown)
    if more:
        listing += f"\n  ... ({more} more)"
    return (f"Extracted {len(landed)} file(s) into {dest_rel}. run_python sees "
            f"them at relative paths:\n{listing}")


SANDBOX_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Run Python offline; cwd is sandbox_io/, its only readable "
                "and writable workspace folder, kept between calls."
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
    {
        "type": "function",
        "function": {
            "name": "extract_archive",
            "description": "Unpack a workspace zip or tar archive into sandbox_io/<name>/ for run_python.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the archive."},
                },
                "required": ["path"],
            },
        },
    },
]

SANDBOX_TOOL_REGISTRY = {"run_python": run_python, "extract_archive": extract_archive}
