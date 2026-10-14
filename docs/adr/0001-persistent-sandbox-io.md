# 1. Persistent `sandbox_io/` as the sandbox's only writable path

Status: accepted

## Context

`run_python` ran in a fresh temp dir per call, and the Seatbelt profile allowed
writes only there. The sandbox could not read any workspace file, and nothing
survived between calls, so multi-step data work was impossible.

## Decision

`sandbox_io/` inside the workspace is the sandbox's cwd and its only readable
and writable workspace path, and it persists across calls. The generated
script still lives in a per-call temp dir that the profile makes read-only.

Because `sandbox_io/` is inside the workspace, the existing file tools reach
it with no new mechanism: `write_file` stages data in, `read_file` reads
results out. `extract_archive` unpacks zip/tar archives into
`sandbox_io/<stem>/`.

`sandbox_io/` is excluded from shadow-git checkpoints and listed in `.gitignore`.
It is scratch data, so `/undo` does not restore it.

## Consequences

- The kernel still denies writes to `agent.py`, `MEMORY.md`, `notes/`, and
  everything else outside `sandbox_io/`. The boundary that matters is intact.
- Sandboxed code can create symlinks inside `sandbox_io/`. Seatbelt checks the
  resolved path, so the sandbox cannot follow them out, and `contain()`
  resolves before its prefix check, so the file tools cannot either.
- Zip-slip through `run_python` (the agent unpacking with `zipfile` itself) is
  unexploitable whatever the archive contains, because the kernel refuses the
  write. `extract_archive` runs in-process, outside Seatbelt, so its member
  checks (traversal, absolute paths, links, size/count/ratio caps, tar
  `filter="data"`) are its real line of defense there. Moving that extraction
  inside the sandbox would make them defense in depth instead.
