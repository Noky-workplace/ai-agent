#!/usr/bin/env python3
"""
Tests for file-based memory (memory.py).

    python3 -m unittest test_memory -v

memory.py decides what the agent believes about you across every future
session — MEMORY.md is injected into the system prompt on every turn. A bug
here doesn't crash; it quietly teaches the agent something wrong, or refuses
to learn something right. So the dedup and supersede logic, which is pure
string handling, gets tested directly.

Each test points memory.py's paths at a fresh temp directory. The module
computes them at import time from cwd, which is right for the app and wrong
for a test run.
"""

import pathlib
import tempfile
import unittest
from unittest import mock

import memory


class MemoryTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="mem-test-")
        root = pathlib.Path(self._tmp.name).resolve()
        self.mem_file = root / "MEMORY.md"
        self.notes = root / "notes"
        self._patches = [
            mock.patch.object(memory, "WORKSPACE", root),
            mock.patch.object(memory, "MEMORY_FILE", self.mem_file),
            mock.patch.object(memory, "NOTES_DIR", self.notes),
            # Literal-search tests must not reach Ollama; TestSemanticFallback
            # turns it back on with a fake embedder.
            mock.patch.object(memory, "SEMANTIC_FALLBACK", False),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        self._tmp.cleanup()

    def facts(self):
        """The fact bodies currently in MEMORY.md, markers stripped."""
        if not self.mem_file.exists():
            return []
        out = []
        for line in self.mem_file.read_text().splitlines():
            if line.startswith("- "):
                out.append(line[2:].split("  _(")[0].strip())
        return out


class TestContainmentScorer(MemoryTest):

    def test_subsumption_scores_high(self):
        """The case the scorer exists for: one fact strictly more specific."""
        score = memory._containment("CPEG undergrad at HKUST", "CPEG undergrad at a university")
        self.assertGreaterEqual(score, memory._NEAR_DUP_THRESHOLD)

    def test_unrelated_facts_score_zero(self):
        self.assertEqual(memory._containment("likes jazz", "studies CPEG at HKUST"), 0.0)

    def test_single_shared_word_is_not_enough(self):
        """'HKUST' alone in common shouldn't flag two different facts."""
        self.assertEqual(memory._containment("TA at HKUST", "studies CPEG at HKUST"), 0.0)

    def test_stopword_only_fact_never_collides(self):
        self.assertEqual(memory._containment("the user is here", "the user is there"), 0.0)

    def test_case_and_punctuation_ignored(self):
        self.assertGreater(memory._containment("Prefers PYTHON, typed!", "prefers python typed"), 0.9)


class TestRemember(MemoryTest):

    def test_first_fact_creates_file_with_header(self):
        out = memory.remember("Prefers concise answers")
        self.assertTrue(out.startswith("Remembered"))
        self.assertIn("# Agent Memory", self.mem_file.read_text())
        self.assertEqual(self.facts(), ["Prefers concise answers"])

    def test_empty_and_overlong_refused(self):
        self.assertTrue(memory.remember("   ").startswith("Error"))
        self.assertTrue(memory.remember("x" * 401).startswith("Error"))
        self.assertFalse(self.mem_file.exists())

    def test_exact_duplicate_not_added(self):
        memory.remember("Prefers concise answers")
        out = memory.remember("prefers concise answers")
        self.assertTrue(out.startswith("Already in memory"))
        self.assertEqual(len(self.facts()), 1)

    def test_near_duplicate_refused_and_names_the_collision(self):
        memory.remember("CPEG undergrad at a university")
        out = memory.remember("CPEG undergrad at HKUST")
        self.assertIn("overlaps an existing memory", out)
        self.assertIn("CPEG undergrad at a university", out)
        self.assertEqual(self.facts(), ["CPEG undergrad at a university"])

    def test_supersede_replaces_in_place(self):
        memory.remember("CPEG undergrad at a university")
        memory.remember("Prefers concise answers")
        out = memory.remember("CPEG undergrad at HKUST", replaces="CPEG undergrad at a university")
        self.assertTrue(out.startswith("Updated memory"))
        self.assertEqual(self.facts(), ["CPEG undergrad at HKUST", "Prefers concise answers"])

    def test_supersede_with_no_match_is_an_error(self):
        memory.remember("Prefers concise answers")
        out = memory.remember("New fact", replaces="nothing like this")
        self.assertTrue(out.startswith("Error: no existing memory"))
        self.assertEqual(self.facts(), ["Prefers concise answers"])

    def test_distinct_facts_both_kept(self):
        memory.remember("Studies CPEG at HKUST")
        memory.remember("Prefers answers in Traditional Chinese")
        self.assertEqual(len(self.facts()), 2)


class TestLoadMemory(MemoryTest):

    def test_missing_file_is_empty(self):
        self.assertEqual(memory.load_memory(), "")

    def test_oversized_memory_is_truncated_with_a_warning(self):
        self.mem_file.write_text("x" * (memory.MAX_MEMORY_CHARS + 500))
        text = memory.load_memory()
        self.assertIn("prune MEMORY.md", text)
        self.assertLess(len(text), memory.MAX_MEMORY_CHARS + 100)


class TestNotes(MemoryTest):

    def test_create_then_append(self):
        self.assertIn("Created", memory.write_note("Linear Algebra", "Av = λv"))
        self.assertIn("Appended", memory.write_note("linear algebra", "det(A - λI) = 0"))
        text = (self.notes / "linear-algebra.md").read_text()
        self.assertIn("Av = λv", text)
        self.assertIn("## Added", text)

    def test_traversal_in_topic_cannot_escape_notes(self):
        memory.write_note("../../.ssh/id_ed25519", "x")
        written = list(self.notes.rglob("*.md"))
        self.assertEqual(len(written), 1)
        self.assertTrue(written[0].resolve().is_relative_to(self.notes.resolve()))

    def test_topic_with_no_usable_characters_refused(self):
        self.assertTrue(memory.write_note("../..", "x").startswith("Error"))

    def test_search_is_literal_and_case_insensitive(self):
        memory.write_note("eigen", "An Eigenvector satisfies Av = λv")
        memory.write_note("context", "num_ctx was 4096 by default")
        out = memory.search_notes("EIGENVECTOR")
        self.assertIn("notes/eigen.md", out)
        self.assertNotIn("context.md", out)

    def test_empty_query_lists_notes(self):
        memory.write_note("a", "x")
        memory.write_note("b", "y")
        out = memory.search_notes("")
        self.assertIn("2 note(s)", out)

    def test_miss_suggests_rephrasing(self):
        memory.write_note("cnn", "convolutional networks")
        self.assertIn("try a different word", memory.search_notes("transformer"))


class TestRegressions(MemoryTest):
    """Bugs found while writing the suite above. Each failed before the fix."""

    def test_exact_dup_check_ignores_the_header(self):
        """'Keep this short' appears in MEMORY_HEADER. A whole-file substring
        check reported it 'Already in memory' and silently dropped it."""
        memory.remember("Prefers concise answers")
        self.assertTrue(memory.remember("Keep this short").startswith("Remembered"))

    def test_shorter_variant_of_a_fact_is_flagged_not_falsely_confirmed(self):
        """'Lives in Hong Kong' is a substring of an existing, different fact.
        Saying 'Already in memory' tells the model the exact fact is stored.
        The near-duplicate path, which shows the existing text, is the honest
        answer."""
        memory.remember("Lives in Hong Kong Island East")
        out = memory.remember("Lives in Hong Kong")
        self.assertFalse(out.startswith("Already in memory"), out)
        self.assertIn("overlaps an existing memory", out)

    def test_updated_marker_does_not_leak_into_comparisons(self):
        """After a supersede the line ends '_(updated YYYY-MM-DD)_'. Only the
        '_(learned ...)_' marker was stripped, so today's date became three
        'content words' and any fact mentioning today collided with it."""
        import datetime
        memory.remember("Works as a TA on Mondays")
        memory.remember("Prefers dark mode", replaces="Works as a TA on Mondays")
        out = memory.remember(f"Exam on {datetime.date.today().isoformat()}")
        self.assertTrue(out.startswith("Remembered"), out)

    def test_updated_line_can_be_superseded_by_exact_text(self):
        memory.remember("Works as a TA on Mondays")
        memory.remember("Works as a TA on Tuesdays", replaces="Works as a TA on Mondays")
        out = memory.remember("Works as a TA on Fridays", replaces="Works as a TA on Tuesdays")
        self.assertIn("was: Works as a TA on Tuesdays", out)
        self.assertNotIn("_(updated", out)   # the echo shows the fact, not the marker
        self.assertEqual(self.facts(), ["Works as a TA on Fridays"])

    def test_ambiguous_replaces_is_refused(self):
        """replaces='HKUST' matched the FIRST line containing it and silently
        overwrote it, though two lines matched. Same rule as edit_file:
        refuse an ambiguous edit rather than guess."""
        memory.remember("Studies CPEG at HKUST")
        memory.remember("Works as a TA at HKUST on Mondays")
        out = memory.remember("Studies Mathematics at HKUST", replaces="HKUST")
        self.assertIn("matches 2 memories", out)
        self.assertEqual(self.facts(), ["Studies CPEG at HKUST", "Works as a TA at HKUST on Mondays"])

    def test_unique_substring_replaces_still_works(self):
        memory.remember("Studies CPEG at HKUST")
        memory.remember("Prefers concise answers")
        out = memory.remember("Studies CPEG and Mathematics at HKUST", replaces="CPEG")
        self.assertTrue(out.startswith("Updated memory"), out)
        self.assertEqual(self.facts()[0], "Studies CPEG and Mathematics at HKUST")

    def test_exact_replaces_wins_over_substring_matches(self):
        """An exact match is unambiguous even if the text also appears inside
        another, longer fact."""
        # Written directly: remember() would (correctly) refuse the second as
        # an overlap of the first, but files edited by hand can hold both.
        self.mem_file.write_text("# Agent Memory\n\n"
                                 "- Studies CPEG  _(learned 2026-01-01)_\n"
                                 "- Studies CPEG and Mathematics at HKUST  _(learned 2026-01-01)_\n")
        out = memory.remember("Studies CPEG at HKUST", replaces="Studies CPEG")
        self.assertTrue(out.startswith("Updated memory"), out)
        self.assertEqual(self.facts(), ["Studies CPEG at HKUST",
                                        "Studies CPEG and Mathematics at HKUST"])

    def test_chinese_note_topic_is_accepted(self):
        """The topic filter kept only ASCII letters, so '線性代數' became an
        empty name and was refused outright."""
        out = memory.write_note("線性代數", "特徵向量滿足 Av = λv")
        self.assertIn("Created", out)
        self.assertTrue((self.notes / "線性代數.md").exists())

    def test_mixed_topic_and_traversal_still_blocked(self):
        memory.write_note("MATH 3332 第一週", "x")
        self.assertTrue((self.notes / "math-3332-第一週.md").exists())
        memory.write_note("../../秘密/../.ssh", "x")
        for p in self.notes.rglob("*.md"):
            self.assertTrue(p.resolve().is_relative_to(self.notes.resolve()))
            self.assertNotIn("/", p.name)

    def test_chinese_facts_are_deduplicated(self):
        """_content_words matched only [a-z0-9], so a Chinese fact had zero
        content words and the near-duplicate check could never fire."""
        memory.remember("喜歡用繁體中文回答")
        out = memory.remember("喜歡用繁體中文回答問題")
        self.assertIn("overlaps an existing memory", out)

    def test_unrelated_chinese_facts_both_kept(self):
        memory.remember("喜歡用繁體中文回答")
        self.assertTrue(memory.remember("在科大讀電腦工程").startswith("Remembered"))

    def test_mixed_language_words(self):
        words = memory._content_words("TA for MATH 3332 課程")
        self.assertTrue({"ta", "math", "3332", "課程"} <= words)


_TOPICS = [("cnn", "convolutional", "image", "vision"),
           ("eigen", "matrix", "linear"),
           ("sharpe", "trading", "backtest")]


def fake_embed(texts):
    """Deterministic stand-in for Ollama: one dimension per topic group."""
    fake_embed.calls.append(list(texts))
    return [[float(sum(t.lower().count(w) for w in group)) + 0.01 for group in _TOPICS]
            for t in texts]


fake_embed.calls = []


class TestSemanticFallback(MemoryTest):

    def setUp(self):
        super().setUp()
        fake_embed.calls.clear()
        for p in (mock.patch.object(memory, "SEMANTIC_FALLBACK", True),
                  mock.patch("semantic.ollama_embed", fake_embed)):
            p.start()
            self._patches.append(p)
        memory.write_note("convolutional networks", "Convolutional filters slide over the image.")
        memory.write_note("linear algebra", "Eigenvalues of a symmetric matrix are real.")
        memory.write_note("backtest", "Sharpe 0.84 for the trading strategy.")

    def test_grep_hit_never_calls_embeddings(self):
        out = memory.search_notes("Sharpe")
        self.assertIn("notes/backtest.md", out)
        self.assertNotIn("semantic", out)
        self.assertEqual(fake_embed.calls, [])

    def test_grep_miss_falls_back_to_semantic_ranking(self):
        out = memory.search_notes("CNN")
        self.assertIn("semantic match", out)
        first = out.index("notes/convolutional-networks.md")
        self.assertLess(first, out.index("notes/linear-algebra.md"))
        self.assertIn("not literal", out)

    def test_unchanged_notes_are_embedded_once(self):
        memory.search_notes("CNN")
        memory.search_notes("vision")
        second = fake_embed.calls[2:]
        self.assertEqual(len(second), 1)                 # only the query
        self.assertTrue(second[0][0].startswith("search_query: "))

    def test_edited_note_is_re_embedded(self):
        memory.search_notes("CNN")
        path = self.notes / "linear-algebra.md"
        path.write_text("Eigen decomposition, edited.")
        stat = path.stat()
        import os
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))
        fake_embed.calls.clear()
        memory.search_notes("CNN")
        docs = [t for batch in fake_embed.calls for t in batch if t.startswith("search_document: ")]
        self.assertEqual(docs, ["search_document: Eigen decomposition, edited."])

    def test_ollama_failure_degrades_to_literal_miss(self):
        def down(texts):
            raise OSError("connection refused")
        with mock.patch("semantic.ollama_embed", down):
            out = memory.search_notes("CNN")
        self.assertIn("try a different word", out)

    def test_cache_is_written_under_agent_cache(self):
        memory.search_notes("CNN")
        self.assertTrue((memory.WORKSPACE / ".agent-cache" / "notes-embeddings.json").exists())

    def test_agent_cache_is_protected_from_file_tools(self):
        import files
        with mock.patch.object(files, "WORKSPACE", memory.WORKSPACE):
            with self.assertRaises(files.Refused):
                files.contain(".agent-cache/notes-embeddings.json")


if __name__ == "__main__":
    unittest.main(verbosity=2)
