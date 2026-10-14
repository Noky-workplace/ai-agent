"""
v2 — File-based memory (the Claude Code approach).

Why files instead of a vector database:

  - You can READ your agent's memory. `cat MEMORY.md`. Debugging a vector
    store means inspecting float arrays and hoping cosine similarity did
    something sensible.
  - You can EDIT it. Wrong fact? Open the file, fix the line. With embeddings
    you re-index.
  - You can VERSION it. git diff shows exactly what your agent learned.
  - No embedding model, no chunking strategy, no similarity threshold to tune.

The retrieval strategy is grep-then-read, which is what Claude Code does with
source files: search for a keyword, get filenames plus matching lines, then
read the promising file in full. Literal substring matching finds exactly what
you asked for and nothing else, which for a personal notes folder of tens or
hundreds of files beats semantic search on predictability.

The tradeoff, stated honestly: grep misses synonyms. A note titled
"convolutional networks" will not surface for the query "CNN" unless that
string appears in it. v8 adds a fallback for exactly that case: when grep
finds nothing, semantic.py ranks notes by embedding similarity and returns
the top few, labelled as semantic. Grep still runs first and its hits are
never replaced (docs/adr/0004-grep-first-semantic-fallback.md).

Layout:
    MEMORY.md      durable facts, injected into the system prompt every turn
    notes/*.md     longer per-topic files, retrieved on demand via search
"""

import datetime
import pathlib
import re
import unicodedata

import semantic

WORKSPACE = pathlib.Path.cwd().resolve()
MEMORY_FILE = WORKSPACE / "MEMORY.md"
NOTES_DIR = WORKSPACE / "notes"

# MEMORY.md is injected into EVERY request, so it costs context on every turn.
# Cap it, or a bloated memory file will crowd out the actual conversation.
MAX_MEMORY_CHARS = 3000
MAX_SEARCH_HITS = 8
SNIPPET_CHARS = 160
# When grep finds nothing, rank notes by embedding similarity (semantic.py).
# Off = v7 behaviour; the notes-retrieval eval flips it to measure the gain.
SEMANTIC_FALLBACK = True
CACHE_FILE = ".agent-cache/notes-embeddings.json"

MEMORY_HEADER = """# Agent Memory

Durable facts about the user, injected into the system prompt every turn.
Keep this short — it costs context on every request. Long-form content
belongs in notes/ instead.
"""

# --------------------------------------------------------------------------
# Near-duplicate detection
# --------------------------------------------------------------------------
# The problem: "CPEG undergrad at HKUST" and "CPEG undergrad at a university"
# are not string-identical, so substring matching misses them. But they aren't
# contradictory either — one is strictly more specific. So the right frame is
# not "reject duplicates", it's "supersede".
#
# We measure CONTAINMENT (shared words / smaller set) rather than Jaccard,
# because containment is high exactly when one fact subsumes another, which is
# the case we care about. Jaccard would score that pair only ~0.4 and miss it.
#
# This is deliberately dumb string logic, not embeddings: it is inspectable,
# has no dependencies, and fails in ways you can predict. The model does the
# actual judging — the code just flags the collision.

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "at", "of", "in", "on",
    "to", "and", "or", "for", "with", "that", "this", "it", "as", "by",
    "user", "users", "their", "they", "his", "her", "he", "she",
}

_NEAR_DUP_THRESHOLD = 0.6
_MIN_SHARED_WORDS = 2

# Every fact line ends in "_(learned DATE)_" or, after a supersede,
# "_(updated DATE)_". Both must be stripped before comparing, or the date
# becomes three "words" (2026, 10, 15) shared with any fact mentioning today.
_MARKER_RE = re.compile(r"\s*_\((?:learned|updated) [^)]*\)_\s*$")

# Chinese and Japanese have no spaces between words, so a word regex sees one
# long run. Character bigrams are the standard cheap substitute for a
# segmenter: "繁體中文" -> 繁體, 體中, 中文. Hangul is included for the same
# reason, though it does use spaces.
_TOKEN_RE = re.compile(r"[a-z0-9]+|[぀-ヿ㐀-䶿一-鿿가-힯]+")
_CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")


def _norm(text: str) -> str:
    """Comparison form: NFC, whitespace collapsed, case-folded."""
    return " ".join(unicodedata.normalize("NFC", text).split()).casefold()


def _fact_lines(existing: str) -> list[tuple[int, str]]:
    """(line index, fact text) for every '- ' line, markers stripped."""
    out = []
    for i, line in enumerate(existing.splitlines()):
        stripped = line.strip()
        if stripped.startswith("- "):
            out.append((i, _MARKER_RE.sub("", stripped[2:]).strip()))
    return out


def _content_words(text: str) -> set[str]:
    """Lowercase, strip punctuation, drop stopwords; CJK runs become bigrams."""
    words = set()
    for tok in _TOKEN_RE.findall(unicodedata.normalize("NFC", text).lower()):
        if _CJK_RE.match(tok):
            words.update(tok[i:i + 2] for i in range(len(tok) - 1))
        elif tok not in _STOPWORDS and len(tok) > 1:
            words.add(tok)
    return words


def _containment(a: str, b: str) -> float:
    """Shared content words divided by the size of the smaller set.

    Returns 0.0 if either side has no content words, so trivial facts like
    "the user is here" never trigger a false collision.
    """
    wa, wb = _content_words(a), _content_words(b)
    if not wa or not wb:
        return 0.0
    shared = wa & wb
    if len(shared) < _MIN_SHARED_WORDS:
        return 0.0
    return len(shared) / min(len(wa), len(wb))


def _find_near_duplicate(fact: str, existing: str) -> str | None:
    """Return the most similar existing memory line, or None."""
    best_line, best_score = None, 0.0
    for _, body in _fact_lines(existing):
        score = _containment(fact, body)
        if score > best_score:
            best_line, best_score = body, score
    return best_line if best_score >= _NEAR_DUP_THRESHOLD else None


def _safe_note_path(topic: str) -> pathlib.Path | None:
    """Turn a topic name into a safe path inside notes/.

    Keeps letters and digits in any script (\\w is Unicode-aware), plus space,
    dash and underscore. Dots, slashes and backslashes are all outside that
    set, so a model asking for '../../.ssh/id_ed25519' gets a harmless
    filename instead — while '線性代數' stays a usable name.
    """
    topic = unicodedata.normalize("NFC", topic)
    stem = re.sub(r"[^\w -]", "", topic).strip().replace(" ", "-").lower()
    if not stem:
        return None
    path = (NOTES_DIR / f"{stem}.md").resolve()
    if not path.is_relative_to(NOTES_DIR.resolve()):
        return None
    return path


def load_memory() -> str:
    """Read MEMORY.md for injection into the system prompt. Called at startup."""
    if not MEMORY_FILE.exists():
        return ""
    text = MEMORY_FILE.read_text(encoding="utf-8", errors="replace")
    if len(text) > MAX_MEMORY_CHARS:
        text = text[:MAX_MEMORY_CHARS] + "\n...[memory truncated — prune MEMORY.md]"
    return text


def remember(fact: str, replaces: str = "") -> str:
    """Append a durable fact to MEMORY.md.

    If `replaces` is given, the matching existing line is overwritten instead
    of a new one being appended. That is how the agent supersedes a vaguer
    fact with a more specific one ("undergrad at a university" ->
    "undergrad at HKUST") rather than accumulating both.
    """
    fact = fact.strip()
    if not fact:
        return "Error: nothing to remember."
    if len(fact) > 400:
        return "Error: too long for MEMORY.md. Use write_note for long content."

    if not MEMORY_FILE.exists():
        MEMORY_FILE.write_text(MEMORY_HEADER, encoding="utf-8")

    existing = MEMORY_FILE.read_text(encoding="utf-8", errors="replace")
    stamp = datetime.date.today().isoformat()

    facts = _fact_lines(existing)

    # --- supersede path ---------------------------------------------------
    # An exact match wins. Otherwise the target may be a fragment, but only
    # if exactly one fact contains it: overwriting the first of several
    # matches is a silent wrong edit, the same failure edit_file refuses.
    if replaces.strip():
        target = _norm(replaces)
        matches = [(i, b) for i, b in facts if _norm(b) == target]
        if not matches:
            matches = [(i, b) for i, b in facts if target in _norm(b)]
        if not matches:
            return (
                f"Error: no existing memory matching {replaces.strip()!r}. "
                f"Call remember with just the fact if you meant to add it as new."
            )
        if len(matches) > 1:
            listing = "\n".join(f"  - {b}" for _, b in matches[:5])
            return (
                f"Error: {replaces.strip()!r} matches {len(matches)} memories:\n{listing}\n"
                f"Nothing was changed. Call again with replaces set to the full text "
                f"of the one to update."
            )
        i, body = matches[0]
        lines = existing.splitlines()
        lines[i] = f"- {fact}  _(updated {stamp})_"
        MEMORY_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return f"Updated memory.\n  was: {body}\n  now: {fact}"

    # --- exact duplicate --------------------------------------------------
    # Compared against fact lines only. A whole-file substring check matched
    # text in the header, and reported a shorter fact as 'already in memory'
    # when only a longer, different fact contained it.
    if any(_norm(body) == _norm(fact) for _, body in facts):
        return f"Already in memory: {fact}"

    # --- near duplicate: refuse, and hand the decision to the model --------
    similar = _find_near_duplicate(fact, existing)
    if similar is not None:
        return (
            f"Not saved — this looks like it overlaps an existing memory:\n"
            f'  existing: "{similar}"\n'
            f'  new:      "{fact}"\n'
            f"If the new fact supersedes the old one, call remember again with "
            f'replaces="{similar}". If both are genuinely separate facts, '
            f"rephrase the new one so it does not restate the old."
        )

    with MEMORY_FILE.open("a", encoding="utf-8") as fh:
        fh.write(f"\n- {fact}  _(learned {stamp})_")

    return f"Remembered: {fact}"


def write_note(topic: str, content: str) -> str:
    """Create or append to notes/<topic>.md for longer-form content."""
    path = _safe_note_path(topic)
    if path is None:
        return f"Error: invalid topic name {topic!r}"

    NOTES_DIR.mkdir(exist_ok=True)
    stamp = datetime.date.today().isoformat()

    if path.exists():
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"\n\n## Added {stamp}\n\n{content.strip()}\n")
        return f"Appended to note: {path.name}"

    path.write_text(
        f"# {topic.strip()}\n\n_Created {stamp}_\n\n{content.strip()}\n",
        encoding="utf-8",
    )
    return f"Created note: {path.name}"


def search_notes(query: str = "") -> str:
    """Grep notes/ for a literal substring. Empty query lists all notes.

    Returns filenames plus matching lines — enough for the model to decide
    which note is worth reading in full with read_file. This two-step
    (search, then read) is the whole retrieval strategy.
    """
    if not NOTES_DIR.exists():
        return "No notes yet. Use write_note to create one."

    files = sorted(NOTES_DIR.glob("*.md"))
    if not files:
        return "No notes yet. Use write_note to create one."

    if not query.strip():
        listing = "\n".join(f"- notes/{f.name}" for f in files)
        return f"{len(files)} note(s):\n{listing}\n\nRead one with read_file."

    needle = query.lower()
    hits = []
    for path in files:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        matches = [ln.strip() for ln in lines if needle in ln.lower()]
        if matches:
            hits.append((path.name, matches[:3]))

    if not hits:
        semantic_out = _semantic_fallback(query, files) if SEMANTIC_FALLBACK else None
        if semantic_out:
            return semantic_out
        available = ", ".join(f.name for f in files)
        return (
            f"No note contains {query!r}. "
            f"Note that this is literal text matching, so try a different word. "
            f"Available notes: {available}"
        )

    out = [f"Notes matching {query!r}:"]
    for name, matches in hits[:MAX_SEARCH_HITS]:
        out.append(f"\nnotes/{name}")
        for m in matches:
            snippet = m[:SNIPPET_CHARS] + ("..." if len(m) > SNIPPET_CHARS else "")
            out.append(f"  | {snippet}")
    out.append("\nRead a full note with read_file, e.g. read_file('notes/xyz.md')")
    return "\n".join(out)


def _semantic_fallback(query: str, files: list[pathlib.Path]) -> str | None:
    """Top notes by meaning, for a query grep missed. None if unavailable.

    Any failure (Ollama down, embedding model not pulled) returns None, so
    search_notes degrades to the plain literal-miss message.
    """
    try:
        ranked = semantic.rank(query, files, WORKSPACE / CACHE_FILE)
    except Exception:
        return None
    if not ranked:
        return None
    out = [f"No note contains {query!r} literally. Semantic matches "
           f"(similar meaning, not literal; read one to check):"]
    for path, score in ranked:
        first = ""
        try:
            first = next((ln.strip() for ln in path.read_text(
                encoding="utf-8", errors="replace").splitlines() if ln.strip()), "")
        except Exception:
            pass
        out.append(f"\nnotes/{path.name}  (semantic match, score {score:.2f})")
        if first:
            snippet = first[:SNIPPET_CHARS] + ("..." if len(first) > SNIPPET_CHARS else "")
            out.append(f"  | {snippet}")
    out.append("\nRead a full note with read_file, e.g. read_file('notes/xyz.md')")
    return "\n".join(out)


MEMORY_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": (
                "Save one short lasting fact about the user; long content goes in write_note."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string", "description": "One short sentence."},
                    "replaces": {
                        "type": "string",
                        "description": (
                            "Existing memory this updates, if any."
                        ),
                    },
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_note",
            "description": (
                "Save summaries, lecture notes or other long content to a topic note; facts use remember."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Short topic name, e.g. 'linear algebra'"},
                    "content": {"type": "string", "description": "Markdown content."},
                },
                "required": ["topic", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": (
                "Search saved notes by literal keyword; an empty query lists all notes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keyword, or empty to list all."}
                },
                "required": [],
            },
        },
    },
]

MEMORY_TOOL_REGISTRY = {
    "remember": remember,
    "write_note": write_note,
    "search_notes": search_notes,
}
