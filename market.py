"""
v7 — Market data for finance mode.

    get_prices(ticker, start, end="", interval="1d")

Fetches OHLCV history and saves it as a CSV inside sandbox_io/data/, then
returns a one-paragraph summary and the path run_python should open.

THE BOUNDARY THIS RESPECTS

run_python runs under a Seatbelt profile with no network. That is the point
of it: model-written code computes on data, it never phones out. So the
network fetch happens HERE, in the agent process, with no model-written code
involved — and the only thing that crosses into the sandbox is a file.
Model code can read market data; it still cannot reach the internet.

No approval prompt, deliberately. Like fetch_url it only reads public data,
and the one thing it writes is a CSV whose path is built from a validated
ticker into the sandbox's scratch directory. Nothing the model controls can
choose where that file lands: the ticker regex has no '/' or '..'.

DATA SOURCE

Yahoo Finance via yfinance: free, no API key, covers HKEX (0700.HK) and the
Hang Seng (^HSI). It is also an unofficial scraper that breaks when Yahoo
changes its site, and its adjusted prices are not audited. Good enough for
research and a portfolio backtest; not good enough to trade on. The tool says
so in every result, because the model should too.

Requires: pip install yfinance  (imported lazily; the rest of the agent
works without it)
"""

import csv
import datetime
import os
import pathlib
import re

TICKER_RE = re.compile(r"^\^?[A-Z0-9][A-Z0-9.\-=]{0,14}$")
INTERVALS = ("1d", "1wk", "1mo")   # intraday omitted: Yahoo caps its history at 60 days
MAX_YEARS = 30
COLUMNS = ("date", "open", "high", "low", "close", "adj_close", "volume")
SOURCE_NOTE = ("Source: Yahoo Finance via yfinance (unofficial; may have gaps or "
               "adjustment errors — fine for research, not for trading).")


def _io_dir() -> pathlib.Path:
    """sandbox_io/, wherever sandbox.py says it is.

    Resolved at call time, not import time, so tests can point it elsewhere
    and so this follows sandbox.IO_DIR if v6 named or moved it.
    """
    try:
        from sandbox import IO_DIR  # type: ignore[attr-defined]
        return pathlib.Path(IO_DIR)
    except Exception:
        import files
        return pathlib.Path(files.WORKSPACE) / "sandbox_io"


def _parse_date(value: str, name: str) -> datetime.date:
    try:
        return datetime.date.fromisoformat(str(value).strip())
    except ValueError:
        raise ValueError(f"{name} must be YYYY-MM-DD, got {value!r}") from None


def _slug(ticker: str) -> str:
    """A filename-safe form: '^HSI' -> 'HSI.idx', '0700.HK' unchanged."""
    if ticker.startswith("^"):
        ticker = ticker[1:] + ".idx"
    return re.sub(r"[^A-Za-z0-9.\-]", "_", ticker)


def _download(ticker: str, start: datetime.date, end: datetime.date,
              interval: str) -> list[dict]:
    """Fetch rows from Yahoo. The only function here that touches the network.

    Isolated so tests replace it, and so swapping data vendors later means
    rewriting one function. yfinance's `end` is exclusive, hence the +1 day.
    """
    import yfinance as yf  # lazy: optional dependency

    df = yf.Ticker(ticker).history(
        start=start.isoformat(),
        end=(end + datetime.timedelta(days=1)).isoformat(),
        interval=interval,
        auto_adjust=False,   # keep raw Close AND Adj Close; the model should see both
        actions=False,
        raise_errors=True,
    )
    rows = []
    for ts, r in df.iterrows():
        rows.append({
            "date": ts.date().isoformat(),
            "open": float(r["Open"]),
            "high": float(r["High"]),
            "low": float(r["Low"]),
            "close": float(r["Close"]),
            "adj_close": float(r.get("Adj Close", r["Close"])),
            "volume": int(r["Volume"]) if r["Volume"] == r["Volume"] else 0,  # NaN-safe
        })
    return rows


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    """Atomic: write a sibling temp file, then rename over the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in COLUMNS})
    os.replace(tmp, path)


def _read_csv(path: pathlib.Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _summary(ticker: str, interval: str, rows: list[dict], rel: str, cached: bool) -> str:
    first, last = rows[0], rows[-1]
    closes = [float(r["close"]) for r in rows]
    change = (closes[-1] / closes[0] - 1) * 100 if closes[0] else 0.0
    return (
        f"{'Loaded cached' if cached else 'Saved'} {len(rows)} {interval} bars of {ticker}, "
        f"{first['date']} to {last['date']}.\n"
        f"First close {closes[0]:.2f}, last close {closes[-1]:.2f} ({change:+.1f}% over the period), "
        f"range {min(closes):.2f}–{max(closes):.2f}.\n"
        f"File: sandbox_io/{rel}\n"
        f"In run_python, open it as '{rel}' (run_python's working directory is sandbox_io). "
        f"Columns: {', '.join(COLUMNS)}.\n"
        f"{SOURCE_NOTE}"
    )


def get_prices(ticker: str, start: str, end: str = "", interval: str = "1d") -> str:
    """Fetch OHLCV history into sandbox_io/data/ and summarise it."""
    ticker = (ticker or "").strip().upper()
    if not TICKER_RE.match(ticker):
        return (f"Error: {ticker!r} is not a valid ticker. HK stocks are a 4-digit code "
                f"plus .HK (0700.HK), US stocks plain (AAPL), indices start with ^ (^HSI).")
    interval = (interval or "1d").strip()
    if interval not in INTERVALS:
        return f"Error: interval must be one of {', '.join(INTERVALS)}."

    today = datetime.date.today()
    try:
        start_d = _parse_date(start, "start")
        end_d = _parse_date(end, "end") if str(end or "").strip() else today
    except ValueError as exc:
        return f"Error: {exc}"
    end_d = min(end_d, today)
    if start_d >= end_d:
        return f"Error: start ({start_d}) must be before end ({end_d})."
    if (end_d - start_d).days > MAX_YEARS * 366:
        return f"Error: range longer than {MAX_YEARS} years."

    rel = f"data/{_slug(ticker)}_{interval}_{start_d}_{end_d}.csv"
    path = _io_dir() / rel

    # A range that ends before today can't change, so a saved file is final.
    # Ranges ending today are refetched: today's bar may still be moving.
    if path.exists() and end_d < today:
        try:
            rows = _read_csv(path)
            if rows:
                return _summary(ticker, interval, rows, rel, cached=True)
        except Exception:
            pass  # corrupt cache: fall through and refetch

    try:
        rows = _download(ticker, start_d, end_d, interval)
    except ImportError:
        return "Error: yfinance is not installed. Run: pip install yfinance"
    except Exception as exc:
        return f"Error: download failed for {ticker} ({str(exc)[:160]})."

    if not rows:
        return (f"No data for {ticker} between {start_d} and {end_d}. Check the ticker — "
                f"HK stocks need .HK (0700.HK) — and that the market was open in that range.")

    _write_csv(path, rows)
    return _summary(ticker, interval, rows, rel, cached=False)


MARKET_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_prices",
            "description": (
                "Download daily/weekly/monthly price history for a stock or index "
                "into a CSV that run_python can read."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string",
                               "description": "e.g. 0700.HK, AAPL, ^HSI"},
                    "start": {"type": "string", "description": "YYYY-MM-DD"},
                    "end": {"type": "string",
                            "description": "YYYY-MM-DD, default today"},
                    "interval": {"type": "string", "enum": list(INTERVALS),
                                 "description": "default 1d"},
                },
                "required": ["ticker", "start"],
            },
        },
    },
]

MARKET_TOOL_REGISTRY = {"get_prices": get_prices}
