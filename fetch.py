"""
v4 — URL fetching.

web_search returns snippets. Snippets are enough to decide whether a page
is worth reading and never enough to answer from. This closes that gap.

SECURITY: the URL comes from the model, and the model gets URLs from search
results — untrusted internet content. Two guards:
  - scheme allowlist: http/https only. file:// would read local disk, and
    the workspace containment in tools.read_file does not apply here.
  - private address block: stops the agent being steered into localhost or
    cloud metadata endpoints (169.254.169.254).
"""

import ipaddress
import socket
import urllib.error
import urllib.parse
import urllib.request

TIMEOUT = 15
MAX_PAGE_CHARS = 3000
MAX_BYTES = 2_000_000

HEADERS = {"User-Agent": "local-agent-project/0.4 (student project)"}


def _is_private(hostname: str) -> bool:
    """Resolve and reject loopback, private, and link-local addresses."""
    try:
        infos = socket.getaddrinfo(hostname, None)
    except Exception:
        return True  # cannot resolve -> refuse
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            return True
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return True
    return False


def fetch_url(url: str) -> str:
    """Fetch a web page and return its readable text."""
    url = (url or "").strip()
    if not url:
        return "Error: no URL provided."

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"Error: refused — only http/https URLs allowed, got {parsed.scheme!r}."
    if not parsed.hostname:
        return "Error: no hostname in URL."
    if _is_private(parsed.hostname):
        return "Error: refused — that address is local or private."

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return "Error: beautifulsoup4 not installed. Run: pip install beautifulsoup4"

    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            ctype = resp.headers.get("Content-Type", "")
            if "html" not in ctype and "text" not in ctype:
                return f"Error: not a text page (Content-Type: {ctype})."
            raw = resp.read(MAX_BYTES)
    except urllib.error.HTTPError as exc:
        return f"Error: server returned HTTP {exc.code}."
    except Exception as exc:
        return f"Error: fetch failed ({exc})."

    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator="\n").split())

    if not text:
        return "Error: page had no readable text (likely JavaScript-rendered)."

    truncated = len(text) > MAX_PAGE_CHARS
    if truncated:
        text = text[:MAX_PAGE_CHARS]

    title = (soup.title.string or "").strip() if soup.title else ""
    head = f"Fetched: {title or url}\n{url}\n\n" if title else f"Fetched: {url}\n\n"
    tail = f"\n...[truncated at {MAX_PAGE_CHARS} chars]" if truncated else ""
    return head + text + tail


FETCH_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": (
                "Read the full text of a web page. Use after web_search when "
                "a result looks relevant and the snippet is not enough to "
                "answer. Only works on pages found via search or given by the user."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full http(s) URL."}
                },
                "required": ["url"],
            },
        },
    },
]

FETCH_TOOL_REGISTRY = {"fetch_url": fetch_url}