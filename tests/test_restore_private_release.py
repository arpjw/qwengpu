"""Check interrupted private-release restore safety."""

import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.restore_private_release import resume_files


class ResumeFilesTests(unittest.TestCase):
    def test_verified_file_is_kept_and_partial_file_is_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            good = directory / "good"
            partial = directory / "partial"
            good.write_bytes(b"complete")
            partial.write_bytes(b"partial")
            expected = {
                "good": hashlib.sha256(b"complete").hexdigest(),
                "partial": hashlib.sha256(b"complete part").hexdigest(),
            }

            self.assertEqual(resume_files(directory, expected), {"good"})
            self.assertEqual(good.read_bytes(), b"complete")
            self.assertFalse(partial.exists())

    def test_unexpected_entry_is_rejected_without_removing_partial(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            partial = directory / "partial"
            partial.write_bytes(b"partial")
            (directory / "unrelated").write_bytes(b"user data")

            with self.assertRaisesRegex(ValueError, "unexpected file"):
                resume_files(directory, {"partial": hashlib.sha256(b"done").hexdigest()})
            self.assertEqual(partial.read_bytes(), b"partial")

    def test_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            target = directory / "target"
            target.write_bytes(b"data")
            (directory / "alias").symlink_to(target)
            with self.assertRaisesRegex(ValueError, "non-file"):
                resume_files(directory, {
                    "target": hashlib.sha256(b"data").hexdigest(),
                    "alias": hashlib.sha256(b"data").hexdigest(),
                })


if __name__ == "__main__":
    unittest.main()
