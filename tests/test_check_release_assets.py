"""Check release identity and asset metadata without downloading weights."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.check_release_assets import verify


class CheckReleaseAssetsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.manifest = Path(self.directory.name) / "backup-manifest.json"
        self.checksum = "a" * 64
        self.manifest.write_text(json.dumps({
            "repo": "owner/repo", "tag": "v4", "release_id": 42,
            "package_checksum_digest": self.checksum,
            "parts": [{"asset": "SHA256SUMS", "file": "SHA256SUMS", "offset": 0,
                       "size": 12, "sha256": self.checksum}],
        }))
        self.release = {"id": 42, "tag_name": "v4", "draft": True}
        self.repository = {"full_name": "owner/repo", "private": True}
        self.assets = [
            {"name": "SHA256SUMS", "size": 12, "digest": f"sha256:{self.checksum}"},
            {"name": "backup-manifest.json", "size": self.manifest.stat().st_size,
             "digest": "sha256:" + hashlib.sha256(self.manifest.read_bytes()).hexdigest()},
        ]

    def test_exact_private_draft(self):
        self.assets[1]["name"] = "corrected-backup-manifest.json"
        result = verify(self.manifest, self.release, self.repository, self.assets)
        self.assertTrue(result["all_asset_sizes_and_sha256_match_manifest"])
        self.assertEqual(result["asset_count"], 2)

    def test_changed_asset_is_rejected(self):
        self.assets[0]["digest"] = "sha256:" + "b" * 64
        with self.assertRaisesRegex(ValueError, "release assets differ"):
            verify(self.manifest, self.release, self.repository, self.assets)

    def test_public_mode_requires_published_public_release(self):
        with self.assertRaisesRegex(ValueError, "public release mode"):
            verify(self.manifest, self.release, self.repository, self.assets, public=True)
        self.release["draft"] = False
        self.repository["private"] = False
        self.assertEqual(verify(self.manifest, self.release, self.repository,
                                self.assets, public=True)["asset_count"], 2)


if __name__ == "__main__":
    unittest.main()
