"""Verify package and destination guards before a Hub checkpoint upload."""

import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from scripts.check_checkpoint_package import package_hashes
from scripts.upload_private_model import check_visibility, main, verify_package


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class UploadModelTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.package = root / "package"
        self.package.mkdir()
        shard = b"tiny weight shard"
        (self.package / "model.safetensors").write_bytes(shard)
        (self.package / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"weight": "model.safetensors"}}))
        for name in ("LICENSE", "README.md", "config.json", "generation_config.json",
                     "qwengpu9b.json", "tokenizer.json", "tokenizer_config.json",
                     "environment.txt"):
            (self.package / name).write_text(name)
        build = json.dumps({"files": [], "sha256": {"model.safetensors": digest(shard)}})
        (self.package / "build-manifest.json").write_text(build)
        self.pinned = root / "pinned-manifest.json"
        self.pinned.write_text(build)
        (self.package / "SHA256SUMS").write_text(package_hashes(self.package, self.pinned))
        self.checksum_digest = digest((self.package / "SHA256SUMS").read_bytes())

    def test_package_mutation_is_rejected(self):
        self.assertIn("model.safetensors", verify_package(
            self.package, self.pinned, self.checksum_digest))
        (self.package / "README.md").write_text("altered card")
        with self.assertRaisesRegex(ValueError, "differ from SHA256SUMS"):
            verify_package(self.package, self.pinned, self.checksum_digest)

    def test_visibility_must_be_preexisting(self):
        check_visibility(True, False)
        check_visibility(False, True)
        with self.assertRaisesRegex(ValueError, "already be public"):
            check_visibility(True, True)
        with self.assertRaisesRegex(ValueError, "already be private"):
            check_visibility(False, False)
        with self.assertRaisesRegex(ValueError, "already be private"):
            check_visibility(None, False)

    def test_public_upload_verifies_revision_and_file_list(self):
        api = MagicMock()
        api.model_info.side_effect = [
            SimpleNamespace(private=False),
            SimpleNamespace(private=False, sha="uploaded-revision"),
        ]
        api.list_repo_files.side_effect = [
            ["old.txt", ".gitattributes"],
            sorted(path.name for path in self.package.iterdir()) + [".gitattributes"],
        ]
        api.upload_folder.return_value = SimpleNamespace(oid="uploaded-revision")
        argv = ["upload_private_model.py", str(self.package), "owner/model",
                "--public", "--pinned-manifest", str(self.pinned),
                "--expected-checksum-digest", self.checksum_digest]
        with patch.object(sys, "argv", argv), patch.object(sys, "stdin", io.StringIO("token\n")), \
             patch("scripts.upload_private_model.HfApi", return_value=api), \
             patch("sys.stdout", new_callable=io.StringIO) as output:
            main()
        report = json.loads(output.getvalue())
        self.assertFalse(report["private"])
        self.assertEqual(report["revision"], "uploaded-revision")
        self.assertEqual(report["removed_stale_files"], ["old.txt"])
        self.assertEqual(api.upload_folder.call_args.kwargs["delete_patterns"], ["old.txt"])

    def test_public_upload_rejects_private_destination_before_sending(self):
        api = MagicMock()
        api.model_info.return_value = SimpleNamespace(private=True)
        argv = ["upload_private_model.py", str(self.package), "owner/model",
                "--public", "--pinned-manifest", str(self.pinned),
                "--expected-checksum-digest", self.checksum_digest]
        with patch.object(sys, "argv", argv), patch.object(sys, "stdin", io.StringIO("token\n")), \
             patch("scripts.upload_private_model.HfApi", return_value=api), \
             patch("scripts.upload_private_model.verify_package",
                   side_effect=AssertionError("should reject before hashing")), \
             patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit):
            main()
        api.upload_folder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
