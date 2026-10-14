#!/usr/bin/env python3
"""
Tests for extract_archive.

    python3 -m unittest test_sandbox -v

The archives are hostile input in the same way paths are: the model picks
which one to unpack, and the archive's author picks every entry name inside
it. Each test builds a small archive in a throwaway workspace and checks that
the unsafe shapes (zip-slip, absolute paths, symlinks, bombs) are refused
before anything is written, and that a normal archive lands where run_python
can see it.
"""

import io
import tarfile
import zipfile

import files
import sandbox
from test_files import WorkspaceTest


class TestExtractArchive(WorkspaceTest):

    def make_zip(self, rel, entries):
        """entries: list of (name, bytes) or a ZipInfo/bytes pair."""
        path = self.ws / rel
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, data in entries:
                zf.writestr(name, data)
        return path

    def make_tar(self, rel, members):
        """members: list of TarInfo, with data bytes for regular files."""
        path = self.ws / rel
        with tarfile.open(path, "w:gz") as tf:
            for info, data in members:
                tf.addfile(info, io.BytesIO(data) if data is not None else None)
        return path

    def io_dir(self):
        return self.ws / sandbox.IO_DIR_NAME

    def test_zip_extracts_into_sandbox_io(self):
        self.make_zip("data.zip", [("a.csv", b"1,2\n"), ("sub/b.txt", b"hi")])
        with self.approve():
            out = sandbox.extract_archive("data.zip")
        self.assertIn("Extracted 2 file(s)", out)
        self.assertEqual((self.io_dir() / "data" / "sub" / "b.txt").read_text(), "hi")

    def test_tar_gz_extracts_with_stem(self):
        info = tarfile.TarInfo("x.txt"); info.size = 3
        self.make_tar("pack.tar.gz", [(info, b"abc")])
        with self.approve():
            sandbox.extract_archive("pack.tar.gz")
        self.assertEqual((self.io_dir() / "pack" / "x.txt").read_text(), "abc")

    def test_zip_slip_refused(self):
        self.make_zip("evil.zip", [("../../escape.txt", b"x")])
        with self.approve() as ask:
            out = sandbox.extract_archive("evil.zip")
        self.assertIn("unsafe entry", out)
        ask.assert_not_called()
        self.assertFalse((self.io_dir() / "evil").exists())

    def test_zip_absolute_path_refused(self):
        self.make_zip("abs.zip", [("/etc/passwd", b"x")])
        with self.approve():
            self.assertIn("unsafe entry", sandbox.extract_archive("abs.zip"))

    def test_zip_symlink_entry_refused(self):
        info = zipfile.ZipInfo("link")
        info.external_attr = (0o120777 << 16)
        self.make_zip("ln.zip", [(info, b"/Users")])
        with self.approve():
            self.assertIn("symlink entry", sandbox.extract_archive("ln.zip"))

    def test_zip_bomb_ratio_refused(self):
        self.make_zip("bomb.zip", [("zeros", b"\0" * 5_000_000)])
        with self.approve():
            self.assertIn("compression ratio", sandbox.extract_archive("bomb.zip"))

    def test_small_compressible_file_allowed(self):
        self.make_zip("small.zip", [("zeros", b"\0" * 50_000)])
        with self.approve():
            self.assertIn("Extracted 1", sandbox.extract_archive("small.zip"))

    def test_tar_traversal_refused(self):
        info = tarfile.TarInfo("../up.txt"); info.size = 1
        self.make_tar("t.tar.gz", [(info, b"x")])
        with self.approve():
            self.assertIn("unsafe entry", sandbox.extract_archive("t.tar.gz"))

    def test_tar_symlink_refused(self):
        info = tarfile.TarInfo("link"); info.type = tarfile.SYMTYPE
        info.linkname = "/Users"
        self.make_tar("l.tar.gz", [(info, None)])
        with self.approve():
            self.assertIn("link or special entry", sandbox.extract_archive("l.tar.gz"))

    def test_existing_destination_refused(self):
        self.make_zip("d.zip", [("a", b"1")])
        (self.io_dir() / "d").mkdir(parents=True)
        with self.approve():
            self.assertIn("already exists", sandbox.extract_archive("d.zip"))

    def test_archive_outside_workspace_refused(self):
        with self.approve():
            self.assertIn("refused", sandbox.extract_archive("../../x.zip"))

    def test_denied_writes_nothing(self):
        self.make_zip("n.zip", [("a", b"1")])
        with self.approve(yes=False):
            self.assertTrue(sandbox.extract_archive("n.zip").startswith("DENIED"))
        self.assertFalse((self.io_dir() / "n").exists())

    def test_plan_mode_refuses(self):
        self.make_zip("p.zip", [("a", b"1")])
        files.PLAN_MODE = True
        self.assertTrue(sandbox.extract_archive("p.zip").startswith("REFUSED"))


if __name__ == "__main__":
    import unittest
    unittest.main(verbosity=2)
