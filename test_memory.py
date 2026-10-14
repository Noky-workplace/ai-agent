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


if __name__ == "__main__":
    unittest.main(verbosity=2)
