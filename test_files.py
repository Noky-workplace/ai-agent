#!/usr/bin/env python3
"""
Tests for the file-safety layer.

    python3 -m pytest test_files.py -v
    python3 test_files.py              # works without pytest installed

WHY THESE TESTS AND NOT OTHERS

contain() is the most security-critical function in the project: it is the
only thing standing between a model-chosen path and the rest of the disk. It
is also pure — a string in, a path or an exception out — so it is cheap to
test exhaustively. That combination (high stakes, zero setup) is what makes it
worth testing first.

The symlink case is the one that matters most. Three 2026 CVEs (PraisonAI
CVE-2026-55540, pgAdmin CVE-2026-7819, npm 'compressing' CVE-2026-40931) were
all the same bug: the containment check used string-only path resolution while
the actual open() followed symlinks, so a link inside the workspace pointing
outside passed the check and read the target anyway. test_symlink_escape is
the regression test for exactly that.

Each test monkeypatches files.WORKSPACE to a fresh temp dir. The module reads
it at import time from cwd, which is right for the app and wrong for a test
run, so every test sets it explicitly rather than depending on where pytest
was invoked from.
"""

import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import files


class WorkspaceTest(unittest.TestCase):
    """Base: a throwaway workspace, with approval auto-granted."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="agent-test-")
        # realpath matters here: on macOS /var is a symlink to /private/var,
        # so an unresolved root would never match a resolved child and every
        # test would fail for the wrong reason.
        self.ws = pathlib.Path(os.path.realpath(self._tmp.name))
        self._patch = mock.patch.object(files, "WORKSPACE", self.ws)
        self._patch.start()
        files.PLAN_MODE = False

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()

    def write(self, rel, text):
        p = self.ws / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p

    def approve(self, yes=True):
        """Replace the interactive prompt. Returns the patcher's context."""
        return mock.patch.object(files, "_ask", return_value=yes)

    def no_checkpoint(self):
        """Skip shadow-git in tests: it shells out and is not what we measure."""
        return mock.patch.object(files, "_checkpoint")


class TestContainment(WorkspaceTest):

    def test_plain_relative_path_resolves(self):
        self.write("notes/a.md", "x")
        self.assertEqual(files.contain("notes/a.md"), self.ws / "notes" / "a.md")

    def test_dotdot_escape_refused(self):
        with self.assertRaises(files.Refused):
            files.contain("../../etc/passwd")

    def test_dotdot_inside_then_out_refused(self):
        with self.assertRaises(files.Refused):
            files.contain("notes/../../secrets.txt")

    def test_absolute_path_outside_refused(self):
        with self.assertRaises(files.Refused):
            files.contain("/etc/passwd")

    def test_symlink_escape_refused(self):
        """The CVE class: a link inside the workspace pointing outside.

        abspath-style checks pass this because the string never leaves the
        workspace; only following the link (realpath / Path.resolve) catches
        it. If this test fails, the jail is open.
        """
        outside = pathlib.Path(os.path.realpath(tempfile.gettempdir())) / "agent-test-target.txt"
        outside.write_text("secret", encoding="utf-8")
        try:
            (self.ws / "innocent.txt").symlink_to(outside)
            with self.assertRaises(files.Refused):
                files.contain("innocent.txt")
        finally:
            outside.unlink(missing_ok=True)

    def test_symlinked_directory_escape_refused(self):
        outside_dir = pathlib.Path(os.path.realpath(tempfile.mkdtemp(prefix="agent-target-")))
        (outside_dir / "k.txt").write_text("secret", encoding="utf-8")
        (self.ws / "linkdir").symlink_to(outside_dir)
        with self.assertRaises(files.Refused):
            files.contain("linkdir/k.txt")

    def test_empty_and_null_refused(self):
        for bad in ("", "   ", "a\x00b"):
            with self.assertRaises(files.Refused):
                files.contain(bad)


class TestDenylist(WorkspaceTest):

    def test_env_refused(self):
        with self.assertRaises(files.Refused):
            files.contain(".env")

    def test_env_case_variants_refused(self):
        """APFS is case-insensitive: .ENV and .env are the same file.

        A case-sensitive denylist check would let the model open the file by
        varying case, so the comparison is casefolded.
        """
        for variant in (".ENV", ".Env", ".eNv"):
            with self.assertRaises(files.Refused):
                files.contain(variant)

    def test_denylist_applies_to_nested_components(self):
        with self.assertRaises(files.Refused):
            files.contain("src/.git/config")

    def test_memory_md_refused(self):
        """MEMORY.md is injected into the system prompt every turn.

        A line written there by a model that read a poisoned web page would
        persist across sessions — prompt injection with a foothold. Only the
        remember tool may touch it.
        """
        with self.assertRaises(files.Refused):
            files.contain("MEMORY.md")

    def test_key_files_refused(self):
        for bad in ("deploy.pem", "id_rsa", "server.key"):
            with self.assertRaises(files.Refused):
                files.contain(bad)

    def test_similar_but_allowed(self):
        """The denylist must not be so broad it blocks ordinary work."""
        for ok in ("environment.md", "notes/keynotes.md", "git-guide.md"):
            self.assertTrue(files.contain(ok).is_relative_to(self.ws))


class TestEditFile(WorkspaceTest):

    def test_replaces_unique_string(self):
        self.write("a.py", "x = 1\ny = 2\n")
        with self.approve(), self.no_checkpoint():
            out = files.edit_file("a.py", "x = 1", "x = 99")
        self.assertIn("verified", out)
        self.assertEqual((self.ws / "a.py").read_text(), "x = 99\ny = 2\n")

    def test_refuses_ambiguous_match(self):
        """Two matches and no replace_all: refuse rather than guess.

        Guessing which occurrence to edit is how an agent silently corrupts
        the wrong function. The error names the count so the model can widen
        old_string instead of retrying blind.
        """
        self.write("a.py", "v = 0\nv = 0\n")
        with self.approve(), self.no_checkpoint():
            out = files.edit_file("a.py", "v = 0", "v = 1")
        self.assertIn("2 times", out)
        self.assertEqual((self.ws / "a.py").read_text(), "v = 0\nv = 0\n")

    def test_replace_all_reports_real_count(self):
        self.write("a.py", "v = 0\nv = 0\nv = 0\n")
        with self.approve(), self.no_checkpoint():
            out = files.edit_file("a.py", "v = 0", "v = 1", replace_all=True)
        self.assertIn("3 of 3", out)
        self.assertNotIn("v = 0", (self.ws / "a.py").read_text())

    def test_missing_string_reports_clearly(self):
        self.write("a.py", "x = 1\n")
        with self.approve(), self.no_checkpoint():
            out = files.edit_file("a.py", "nope", "y")
        self.assertIn("not found", out)

    def test_denial_leaves_file_untouched(self):
        self.write("a.py", "x = 1\n")
        with self.approve(False), self.no_checkpoint():
            out = files.edit_file("a.py", "x = 1", "x = 2")
        self.assertIn("DENIED", out)
        self.assertEqual((self.ws / "a.py").read_text(), "x = 1\n")


class TestWriteAndDelete(WorkspaceTest):

    def test_write_then_read_back(self):
        with self.approve(), self.no_checkpoint():
            out = files.write_file("notes/new.md", "hello\n")
        self.assertIn("verified", out)
        self.assertEqual((self.ws / "notes" / "new.md").read_text(), "hello\n")

    def test_write_refuses_outside_workspace(self):
        with self.approve(), self.no_checkpoint():
            out = files.write_file("../escape.txt", "x")
        self.assertIn("refused", out)

    def test_delete_moves_to_trash_not_oblivion(self):
        self.write("gone.txt", "bye")
        with self.approve(), self.no_checkpoint():
            out = files.delete_file("gone.txt")
        self.assertIn("trash", out.lower())
        self.assertFalse((self.ws / "gone.txt").exists())
        self.assertEqual(len(list((self.ws / files.TRASH_DIR).iterdir())), 1)

    def test_oversized_write_refused(self):
        with self.approve(), self.no_checkpoint():
            out = files.write_file("big.txt", "x" * (files.MAX_WRITE_CHARS + 1))
        self.assertIn("too long", out)


class TestPlanMode(WorkspaceTest):
    """Plan mode is a state, not a prompt. It must not even ask."""

    def setUp(self):
        super().setUp()
        files.PLAN_MODE = True

    def tearDown(self):
        files.PLAN_MODE = False
        super().tearDown()

    def test_all_mutations_refused_without_prompting(self):
        self.write("a.txt", "keep")
        # _ask raising proves plan mode short-circuits before any prompt.
        boom = mock.patch.object(files, "_ask", side_effect=AssertionError("prompted"))
        with boom:
            self.assertIn("plan mode", files.write_file("b.txt", "x"))
            self.assertIn("plan mode", files.edit_file("a.txt", "keep", "x"))
            self.assertIn("plan mode", files.delete_file("a.txt"))
        self.assertEqual((self.ws / "a.txt").read_text(), "keep")

    def test_reads_still_work(self):
        self.write("a.txt", "keep")
        self.assertIn("a.txt", files.list_files("."))


if __name__ == "__main__":
    unittest.main(verbosity=2)
