"""
v8 — Semantic fallback for search_notes.

search_notes greps first, always. Only when grep finds nothing does it come
here: embed the query and every note with Ollama's nomic-embed-text, rank by
cosine similarity, and return the top few with their scores, labelled as
semantic so the model knows they are not literal matches.

WHY GREP STAYS FIRST (see docs/adr/0004)

A literal hit is exact and inspectable: the matching line is right there.
A similarity score is neither. So embeddings only answer the question grep
cannot ("CNN" when the note says "convolutional networks"), never replace
an answer grep already gave.

CACHE

Note vectors are cached in .agent-cache/notes-embeddings.json, keyed by
note path + mtime + model, so an unchanged note is embedded once. Only the
query is embedded on each fallback. The cache is derived data: delete it and
it rebuilds.

Needs: ollama pull nomic-embed-text (~270 MB). No Python dependency.
"""

import json
import math
import os
import pathlib
import urllib.request

EMBED_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"
TOP_K = 3
MAX_NOTE_CHARS = 6000       # nomic's context is 2048 tokens in Ollama by default
TIMEOUT_S = 30

# nomic-embed-text was trained with these task prefixes; without them,
# query/document similarity is noticeably worse.
QUERY_PREFIX = "search_query: "
DOC_PREFIX = "search_document: "


def ollama_embed(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    """Embed texts with Ollama's /api/embed. Raises on any failure."""
    body = json.dumps({"model": model, "input": texts}).encode("utf-8")
    req = urllib.request.Request(EMBED_URL, data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    vectors = data.get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        raise RuntimeError(f"unexpected embed response: {str(data)[:120]}")
    return vectors


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _load_cache(path: pathlib.Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}   # missing or corrupt: rebuild


def _save_cache(path: pathlib.Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(cache), encoding="utf-8")
    os.replace(tmp, path)


def rank(query: str, files: list[pathlib.Path], cache_path: pathlib.Path,
         embed=None, model: str = EMBED_MODEL,
         top_k: int = TOP_K) -> list[tuple[pathlib.Path, float]]:
    """Top notes by cosine similarity to the query, best first.

    Embeds only notes whose (path, mtime, model) is not cached. Raises if
    Ollama is unreachable or the model is missing; the caller degrades to
    the plain grep miss.
    """
    embed = embed or ollama_embed
    cache = _load_cache(cache_path)
    keys, todo = {}, []
    for f in files:
        key = str(f)
        stamp = f"{f.stat().st_mtime_ns}:{model}"
        keys[f] = key
        entry = cache.get(key)
        if not (isinstance(entry, dict) and entry.get("stamp") == stamp):
            text = f.read_text(encoding="utf-8", errors="replace")[:MAX_NOTE_CHARS]
            todo.append((f, key, stamp, DOC_PREFIX + text))

    if todo:
        vectors = embed([t[3] for t in todo])
        for (f, key, stamp, _), vec in zip(todo, vectors):
            cache[key] = {"stamp": stamp, "vec": vec}
        # Drop entries for notes that no longer exist.
        live = set(keys.values())
        cache = {k: v for k, v in cache.items() if k in live}
        _save_cache(cache_path, cache)

    qvec = embed([QUERY_PREFIX + query])[0]
    scored = [(f, cosine(qvec, cache[keys[f]]["vec"])) for f in files]
    scored.sort(key=lambda fs: fs[1], reverse=True)
    return scored[:top_k]
