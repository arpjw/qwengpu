"""Exercise public Hub staging promotion without changing a real repo."""

import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from huggingface_hub import CommitOperationCopy, CommitOperationDelete

from scripts.check_checkpoint_package import REQUIRED
from scripts.promote_hf_model import main, parse_checksums, promotion_operations


STAGE_SHA = "a" * 40
MAIN_SHA = "b" * 40
PUBLISHED_SHA = "c" * 40


class PromoteHfModelTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        names = sorted(REQUIRED | {"model.safetensors"})
        self.package_files = set(names) | {"SHA256SUMS"}
        self.checksums = "".join(f"{'d' * 64}  {name}\n" for name in names).encode()
        self.digest = hashlib.sha256(self.checksums).hexdigest()
        self.checksum_path = Path(self.directory.name) / "SHA256SUMS"
        self.checksum_path.write_bytes(self.checksums)

    def test_checksum_list_rejects_unsafe_paths_and_duplicates(self):
        self.assertEqual(parse_checksums(self.checksums.decode()), self.package_files)
        with self.assertRaisesRegex(ValueError, "unsafe"):
            parse_checksums(self.checksums.decode() + f"{'d' * 64}  ../other\n")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_checksums(self.checksums.decode() + f"{'d' * 64}  README.md\n")

    def test_operations_copy_stage_and_delete_only_stale_main_files(self):
        operations, stale = promotion_operations(self.package_files,
                                                 {"README.md", "old.txt", ".gitattributes"},
                                                 STAGE_SHA)
        self.assertEqual(stale, ["old.txt"])
        self.assertEqual(sum(isinstance(op, CommitOperationCopy) for op in operations),
                         len(self.package_files))
        self.assertEqual(sum(isinstance(op, CommitOperationDelete) for op in operations), 1)
        self.assertTrue(all(op.src_revision == STAGE_SHA for op in operations
                            if isinstance(op, CommitOperationCopy)))

    def run_cli(self, *, apply: bool):
        api = MagicMock()
        api.model_info.side_effect = [
            SimpleNamespace(private=False, sha=MAIN_SHA),
            SimpleNamespace(private=False, sha=STAGE_SHA),
            *([SimpleNamespace(private=False, sha=MAIN_SHA),
               SimpleNamespace(private=False, sha=PUBLISHED_SHA)] if apply else []),
        ]
        api.list_repo_files.side_effect = [
            sorted(self.package_files) + [".gitattributes"],
            ["README.md", "old.txt", ".gitattributes"],
            *([sorted(self.package_files) + [".gitattributes"]] if apply else []),
        ]
        api.create_commit.return_value = SimpleNamespace(oid=PUBLISHED_SHA)
        argv = ["promote_hf_model.py", "owner/model", STAGE_SHA, self.digest]
        if apply:
            argv.append("--apply")
        with patch.object(sys, "argv", argv), patch.object(sys, "stdin", io.StringIO("token\n")), \
             patch("scripts.promote_hf_model.HfApi", return_value=api), \
             patch("scripts.promote_hf_model.hf_hub_download",
                   return_value=str(self.checksum_path)), \
             patch("sys.stdout", new_callable=io.StringIO) as output:
            main()
        return json.loads(output.getvalue()), api

    def test_dry_run_never_updates_main(self):
        report, api = self.run_cli(apply=False)
        self.assertFalse(report["applied"])
        self.assertEqual(report["stale_main_files"], ["old.txt"])
        api.create_commit.assert_not_called()

    def test_apply_uses_one_parent_pinned_commit(self):
        report, api = self.run_cli(apply=True)
        self.assertTrue(report["applied"])
        self.assertEqual(report["main_after"], PUBLISHED_SHA)
        self.assertEqual(api.create_commit.call_count, 1)
        self.assertEqual(api.create_commit.call_args.kwargs["parent_commit"], MAIN_SHA)
        self.assertEqual(api.create_commit.call_args.kwargs["revision"], "main")

    def test_apply_rejects_staged_file_list_mismatch(self):
        api = MagicMock()
        api.model_info.side_effect = [
            SimpleNamespace(private=False, sha=MAIN_SHA),
            SimpleNamespace(private=False, sha=STAGE_SHA),
        ]
        api.list_repo_files.return_value = sorted(self.package_files - {"README.md"})
        argv = ["promote_hf_model.py", "owner/model", STAGE_SHA,
                self.digest, "--apply"]
        with patch.object(sys, "argv", argv), patch.object(sys, "stdin", io.StringIO("token\n")), \
             patch("scripts.promote_hf_model.HfApi", return_value=api), \
             patch("scripts.promote_hf_model.hf_hub_download",
                   return_value=str(self.checksum_path)), \
             patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit):
            main()
        api.create_commit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
