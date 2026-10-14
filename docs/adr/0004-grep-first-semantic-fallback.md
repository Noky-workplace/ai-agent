# 4. Notes search: grep first, embeddings only as a fallback

Status: accepted

Follows the v2 decision recorded in `memory.py`'s module docstring (cited
elsewhere as "ADR 001", though it never had its own file): notes are plain
markdown, retrieved by grep-then-read, with "grep misses synonyms" accepted
as the cost.

## Context

That cost is real. "CNN" finds nothing in a note titled "Convolutional
networks", and the model is told only to try a different word.

## Decision

`search_notes` still greps first. Only when grep finds nothing does
`semantic.py` embed the query and every note with Ollama's
`nomic-embed-text`, rank by cosine similarity, and return the top 3 with
scores, labelled "semantic match ... not literal" so the model reads the note
before relying on it.

Grep stays first because a literal hit is exact and shows the matching line;
a similarity score is neither. Embeddings answer only what grep cannot.

- Note vectors are cached in `.agent-cache/notes-embeddings.json`, keyed by
  path, mtime and model, so only the query is embedded on a repeat search.
  `.agent-cache` is git-ignored and on the file tools' denylist.
- If Ollama or the model is unavailable, search_notes returns the old
  literal-miss message. No new Python dependency.

## Evidence

`python3 -m evals.notes_retrieval`: 8 fixture notes, 15 queries (5 literal,
10 synonyms that don't appear in their note)
(`evals/results/20261015-060837-notes-retrieval.json`):

| | Hit (in results) | Expected note first |
|---|---|---|
| grep only | 5/15 (33.3%); synonyms 0/10 | 5/15 |
| grep + fallback | 15/15 (100%); synonyms 10/10 | 14/15; synonyms 9/10 |

Literal queries are unchanged: the fallback never runs when grep hits.

## Limits

- Top 3 out of 8 notes is a generous bar. Similarity scores were close
  (all 0.45–0.67; for "dependency management" the top two tied at 0.53),
  and for "CNN" the right note ranked third behind "transformers". The label and the scores are there so
  the model treats a semantic match as a lead, not an answer.
- A query grep matches in the wrong note never reaches the fallback.
- 15 queries over 8 notes is small. Re-run with a larger fixture before
  changing `TOP_K` or adding a score floor.
