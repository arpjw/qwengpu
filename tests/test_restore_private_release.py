"""Check interrupted private-release restore safety."""

import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.restore_private_release import check_source, main, resume_files


class SourceModeTests(unittest.TestCase):
    def test_private_draft_and_public_release_are_distinct(self):
        release = {"tag_name": "v4", "draft": True}
        repo = {"private": True}
        check_source(release, repo, "v4", False)
        with self.assertRaisesRegex(RuntimeError, "public mode"):
            check_source(release, repo, "v4", True)
        release["draft"] = False
        repo["private"] = False
        check_source(release, repo, "v4", True)
        with self.assertRaisesRegex(RuntimeError, "private mode"):
            check_source(release, repo, "v4", False)

    def test_tag_mismatch_is_rejected_in_both_modes(self):
        for public in (False, True):
            with self.subTest(public=public), self.assertRaisesRegex(RuntimeError, "tag differs"):
                check_source({"tag_name": "wrong", "draft": not public},
                             {"private": not public}, "v4", public)

    def test_public_mode_restores_and_verifies_tiny_package(self):
        content = b"tiny verified content"
        content_digest = hashlib.sha256(content).hexdigest()
        checksums = f"{content_digest}  README.md\n".encode()
        checksum_digest = hashlib.sha256(checksums).hexdigest()
        manifest = json.dumps({
            "repo": "owner/repo", "tag": "v4", "release_id": 42,
            "package_checksum_digest": checksum_digest,
            "parts": [
                {"file": "SHA256SUMS", "asset": "SHA256SUMS", "offset": 0,
                 "size": len(checksums), "sha256": checksum_digest},
                {"file": "README.md", "asset": "README.md", "offset": 0,
                 "size": len(content), "sha256": content_digest},
            ],
        }).encode()
        payloads = {"backup-manifest.json": manifest,
                    "SHA256SUMS": checksums, "README.md": content}

        def download(_repo, _tag, name, directory, _token):
            path = directory / name
            path.write_bytes(payloads[name])
            return path

        def gh(args, _token):
            if args[1].endswith("/releases/42"):
                return json.dumps({"tag_name": "v4", "draft": False})
            return json.dumps({"private": False})

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "restored"
            argv = ["restore_private_release.py", "owner/repo", "v4", "42",
                    checksum_digest, str(destination), "--public",
                    "--backup-manifest", "backup-manifest.json"]
            stdin = io.StringIO("unused token\n")
            with patch.object(sys, "argv", argv), patch.object(sys, "stdin", stdin), \
                 patch("scripts.restore_private_release.run_gh", side_effect=gh), \
                 patch("scripts.restore_private_release.download", side_effect=download), \
                 patch("sys.stdout", new_callable=io.StringIO):
                main()
            self.assertEqual(stdin.tell(), 0)
            self.assertEqual((destination / "README.md").read_bytes(), content)
            self.assertEqual((destination / "SHA256SUMS").read_bytes(), checksums)


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
