"""
v3 — Academic paper search (arXiv + Semantic Scholar).

Both APIs are free and need no key, so this module stays standard-library
only: urllib for the request, xml.etree for arXiv's Atom feed, json for
Semantic Scholar.

Design note — why ONE tool with a `source` parameter instead of two tools:
the model has to pick from every schema on every turn, and an 8B model's
selection accuracy degrades noticeably past ~8 tools. Two paper-search tools
that do almost the same thing is exactly the ambiguity that causes bad picks.
Collapsing them into one tool with a mode moves the choice into a parameter,
where it's a smaller decision on a shorter list.

Which source to use:
  arxiv             preprints, full abstracts, strongest for CS/physics/math,
                    and the place recent ML work appears first
  semantic_scholar  broader coverage across fields, plus citation counts,
                    which is the fastest proxy for whether a paper matters
"""

import json
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ARXIV_API = "http://export.arxiv.org/api/query"
S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"

TIMEOUT = 20
MAX_RESULTS = 5
ABSTRACT_CHARS = 400

# Both APIs ask for a descriptive User-Agent. Semantic Scholar rate-limits
# anonymous traffic harder without one.
HEADERS = {"User-Agent": "local-agent-project/0.3 (student project)"}


def _truncate(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def _search_arxiv(query: str, max_results: int) -> str:
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
    })
    req = urllib.request.Request(f"{ARXIV_API}?{params}", headers=HEADERS)

    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        return f"Error: arXiv returned HTTP {exc.code}."
    except Exception as exc:
        return f"Error: arXiv request failed ({exc})."

    # arXiv speaks Atom, not JSON.
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        return f"Error: could not parse arXiv response ({exc})."

    entries = root.findall("atom:entry", ns)
    if not entries:
        return f"No arXiv papers found for {query!r}."

    out = [f"arXiv results for {query!r}:"]
    for i, entry in enumerate(entries, 1):
        title = _truncate(entry.findtext("atom:title", "", ns), 160)
        summary = _truncate(entry.findtext("atom:summary", "", ns), ABSTRACT_CHARS)
        published = (entry.findtext("atom:published", "", ns) or "")[:10]
        link = entry.findtext("atom:id", "", ns)
        authors = [
            a.findtext("atom:name", "", ns)
            for a in entry.findall("atom:author", ns)
        ][:3]
        author_str = ", ".join(a for a in authors if a)
        if len(entry.findall("atom:author", ns)) > 3:
            author_str += " et al."

        out.append(
            f"\n{i}. {title}\n"
            f"   {author_str} ({published})\n"
            f"   {summary}\n"
            f"   {link}"
        )
    return "\n".join(out)


def _search_semantic_scholar(query: str, max_results: int) -> str:
    params = urllib.parse.urlencode({
        "query": query,
        "limit": max_results,
        "fields": "title,abstract,year,authors,citationCount,url",
    })
    req = urllib.request.Request(f"{S2_API}?{params}", headers=HEADERS)

    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            return ("Error: Semantic Scholar rate limit hit. Wait a minute, "
                    "or use source='arxiv' instead.")
        return f"Error: Semantic Scholar returned HTTP {exc.code}."
    except Exception as exc:
        return f"Error: Semantic Scholar request failed ({exc})."

    papers = data.get("data") or []
    if not papers:
        return f"No Semantic Scholar papers found for {query!r}."

    out = [f"Semantic Scholar results for {query!r}:"]
    for i, paper in enumerate(papers, 1):
        title = _truncate(paper.get("title", "(untitled)"), 160)
        abstract = _truncate(paper.get("abstract") or "(no abstract)", ABSTRACT_CHARS)
        year = paper.get("year") or "n.d."
        cites = paper.get("citationCount", 0)
        names = [a.get("name", "") for a in (paper.get("authors") or [])][:3]
        author_str = ", ".join(n for n in names if n)
        if len(paper.get("authors") or []) > 3:
            author_str += " et al."

        out.append(
            f"\n{i}. {title}\n"
            f"   {author_str} ({year}) — {cites} citations\n"
            f"   {abstract}\n"
            f"   {paper.get('url', '')}"
        )
    return "\n".join(out)


def search_papers(query: str, source: str = "arxiv", max_results: int = MAX_RESULTS) -> str:
    """Search academic papers. source is 'arxiv' or 'semantic_scholar'."""
    query = (query or "").strip()
    if not query:
        return "Error: empty query."

    try:
        max_results = max(1, min(int(max_results), 10))
    except (TypeError, ValueError):
        max_results = MAX_RESULTS

    source = (source or "arxiv").strip().lower().replace("-", "_")
    if source in ("arxiv", "arx"):
        return _search_arxiv(query, max_results)
    if source in ("semantic_scholar", "semanticscholar", "s2"):
        return _search_semantic_scholar(query, max_results)
    return (f"Error: unknown source {source!r}. "
            f"Use 'arxiv' or 'semantic_scholar'.")


RESEARCH_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": (
                "Search academic papers by keyword. Use for any question about "
                "research, papers, studies, or academic literature. "
                "source='arxiv' for CS/physics/math preprints and recent ML work; "
                "source='semantic_scholar' for broader fields and citation counts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Topic keywords, e.g. 'attention transformers'",
                    },
                    "source": {
                        "type": "string",
                        "enum": ["arxiv", "semantic_scholar"],
                        "description": "Which database to search. Default arxiv.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

RESEARCH_TOOL_REGISTRY = {"search_papers": search_papers}
