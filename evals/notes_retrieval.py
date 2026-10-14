#!/usr/bin/env python3
"""
Notes-retrieval eval: grep alone vs grep + semantic fallback.

    python3 -m evals.notes_retrieval            # needs Ollama + nomic-embed-text

Writes the fixture notes from notes_retrieval.json into a temporary
workspace, runs search_notes for each query twice (fallback off, then on),
and scores a hit when the expected note's filename appears in the result:
any grep hit, or one of the semantic top 3. The model is not involved; this
measures retrieval only. Results are saved under evals/results/.

"literal" queries appear verbatim in their note and should be grep hits
either way. "synonym" queries do not, so grep alone misses them; they are
what the fallback exists for.
"""

import argparse
import datetime
import json
import pathlib
import sys
import tempfile
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURE = pathlib.Path(__file__).with_name("notes_retrieval.json")
RESULTS_DIR = pathlib.Path(__file__).with_name("results")


def run(fixture: dict, fallback: bool) -> list[dict]:
    import memory
    records = []
    with tempfile.TemporaryDirectory(prefix="notes-eval-") as tmp:
        root = pathlib.Path(tmp).resolve()
        notes = root / "notes"
        notes.mkdir()
        for name, text in fixture["notes"].items():
            (notes / name).write_text(text, encoding="utf-8")
        with mock.patch.object(memory, "WORKSPACE", root), \
                mock.patch.object(memory, "NOTES_DIR", notes), \
                mock.patch.object(memory, "SEMANTIC_FALLBACK", fallback):
            for q in fixture["queries"]:
                out = memory.search_notes(q["query"])
                semantic = "semantic match" in out
                # First note listed in the result, grep or semantic.
                listed = [ln.strip().split()[0] for ln in out.splitlines()
                          if ln.strip().startswith("notes/")]
                records.append({
                    "id": q["id"], "query": q["query"], "kind": q["kind"],
                    "expect": q["expect"],
                    "hit": f"notes/{q['expect']}" in out,
                    "top1": bool(listed) and listed[0] == f"notes/{q['expect']}",
                    "via": "semantic" if semantic else ("grep" if "Notes matching" in out else "none"),
                    "output": out,
                })
    return records


def summarise(records: list[dict]) -> dict:
    def acc(rs, key="hit"):
        return round(sum(r[key] for r in rs) / len(rs), 4) if rs else None
    return {
        "n": len(records),
        "accuracy": acc(records),
        "top1": acc(records, "top1"),
        "synonym_top1": acc([r for r in records if r["kind"] == "synonym"], "top1"),
        "literal": acc([r for r in records if r["kind"] == "literal"]),
        "synonym": acc([r for r in records if r["kind"] == "synonym"]),
        "misses": [r["id"] for r in records if not r["hit"]],
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", default=str(RESULTS_DIR))
    args = p.parse_args(argv)

    import semantic
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    grep_only = run(fixture, fallback=False)
    with_fallback = run(fixture, fallback=True)
    if not any(r["via"] == "semantic" for r in with_fallback):
        print("warning: the fallback never ran. Is Ollama up, with "
              f"{semantic.EMBED_MODEL} pulled?")

    report = {
        "grep_only": {"summary": summarise(grep_only), "records": grep_only},
        "grep_plus_fallback": {"summary": summarise(with_fallback), "records": with_fallback},
        "meta": {"embed_model": semantic.EMBED_MODEL, "top_k": semantic.TOP_K,
                 "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                 "queries": len(fixture["queries"]), "notes": len(fixture["notes"])},
    }
    for label in ("grep_only", "grep_plus_fallback"):
        s = report[label]["summary"]
        print(f"{label:<20} {s['accuracy']:.1%} (top-1 {s['top1']:.1%})   literal {s['literal']:.1%}   "
              f"synonym {s['synonym']:.1%} (top-1 {s['synonym_top1']:.1%})   "
              f"misses {', '.join(s['misses']) or '-'}")
    for r in with_fallback:
        if r["via"] == "semantic":
            ranked = [ln.split("  (")[0] + " " + ln.split("score ")[1].rstrip(")")
                      for ln in r["output"].splitlines() if "semantic match, score" in ln]
            mark = "ok  " if r["hit"] else "MISS"
            print(f"  [{mark}] {r['id']} {r['query']!r:<42} -> {'; '.join(ranked)}")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = out / f"{stamp}-notes-retrieval.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
