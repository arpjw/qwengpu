"""Check the fixed v4 model-card-only equivalence gate."""

import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.check_v4_package_equivalence import compare, read_checksums


class PackageEquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.original = {f"model-{i:05d}-of-00035.safetensors": "a" * 64
                         for i in range(1, 36)}
        self.original.update({f"file-{i}.json": "b" * 64 for i in range(12)})
        self.original["README.md"] = "c" * 64
        self.corrected = {**self.original, "README.md": "d" * 64}

    def test_accepts_card_only_change(self):
        result = compare(self.original, self.corrected)
        self.assertEqual(result["changed_files"], ["README.md"])
        self.assertEqual(result["unchanged_content_files"], 47)
        self.assertEqual(result["unchanged_weight_shards"], 35)

    def test_rejects_weight_change_or_missing_file(self):
        changed_shard = {**self.corrected,
                         "model-00001-of-00035.safetensors": "e" * 64}
        with self.assertRaisesRegex(ValueError, "only README.md"):
            compare(self.original, changed_shard)
        missing = dict(self.corrected)
        missing.pop("file-0.json")
        with self.assertRaisesRegex(ValueError, "file lists differ"):
            compare(self.original, missing)

    def test_rejects_changed_checksum_list_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "SHA256SUMS"
            contents = b"a" * 64 + b"  README.md\n"
            path.write_bytes(contents)
            expected = hashlib.sha256(contents).hexdigest()
            self.assertEqual(read_checksums(path, expected), {"README.md": "a" * 64})
            with self.assertRaisesRegex(ValueError, "digest differs"):
                read_checksums(path, "0" * 64)


if __name__ == "__main__":
    unittest.main()
