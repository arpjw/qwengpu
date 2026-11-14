"""Check GitHub release asset metadata against a pinned backup manifest.

This does not download checkpoint bytes. A clean restore and package hash
check are still required before treating the model package as verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def gh_json(endpoint: str):
    return json.loads(subprocess.check_output(["gh", "api", endpoint], text=True))


def verify(manifest_path: Path, release: dict, repository: dict,
           assets: list[dict], *, public: bool = False) -> dict:
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if release["id"] != manifest["release_id"] or release["tag_name"] != manifest["tag"]:
        raise ValueError("release identity differs from backup manifest")
    if repository["full_name"] != manifest["repo"]:
        raise ValueError("repository identity differs from backup manifest")
    if public:
        if repository["private"] or release["draft"]:
            raise ValueError("public release mode requires a published public release")
    elif not repository["private"] or not release["draft"]:
        raise ValueError("private draft mode requires a private repository and draft release")

    expected = {}
    for part in manifest["parts"]:
        name = part["asset"]
        if name in expected:
            raise ValueError(f"duplicate backup asset: {name}")
        expected[name] = (part["size"], part["sha256"])
    checksum = expected.get("SHA256SUMS")
    if checksum is None or checksum[1] != manifest["package_checksum_digest"]:
        raise ValueError("backup checksum asset differs from pinned package digest")
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    actual = {}
    for asset in assets:
        name = asset["name"]
        if name in actual:
            raise ValueError(f"duplicate live release asset: {name}")
        digest = asset.get("digest") or ""
        if not digest.startswith("sha256:"):
            raise ValueError(f"missing SHA-256 for release asset: {name}")
        actual[name] = (asset["size"], digest.removeprefix("sha256:"))
    manifest_assets = [name for name, value in actual.items()
                       if name not in expected and value == (len(manifest_bytes), manifest_digest)]
    if len(manifest_assets) != 1:
        raise ValueError("expected exactly one release asset matching the backup manifest")
    expected[manifest_assets[0]] = (len(manifest_bytes), manifest_digest)
    if actual != expected:
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        changed = sorted(name for name in expected.keys() & actual.keys()
                         if expected[name] != actual[name])
        raise ValueError(f"release assets differ: missing={missing}, extra={extra}, changed={changed}")
    return {
        "repo": manifest["repo"],
        "tag": manifest["tag"],
        "release_id": release["id"],
        "private": repository["private"],
        "draft": release["draft"],
        "asset_count": len(actual),
        "asset_bytes": sum(size for size, _ in actual.values()),
        "backup_manifest_sha256": manifest_digest,
        "package_checksum_digest": manifest["package_checksum_digest"],
        "all_asset_sizes_and_sha256_match_manifest": True,
        "qualification": "Metadata check only; see the clean restore and package checker for byte verification.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--public", action="store_true",
                        help="require a public repository and published release")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    repo = manifest["repo"]
    release_id = manifest["release_id"]
    release = gh_json(f"repos/{repo}/releases/{release_id}")
    repository = gh_json(f"repos/{repo}")
    assets = []
    for page in range(1, 100):
        batch = gh_json(f"repos/{repo}/releases/{release_id}/assets?per_page=100&page={page}")
        assets.extend(batch)
        if len(batch) < 100:
            break
    else:
        raise ValueError("release asset list exceeds 9,900 entries")
    report = verify(args.manifest, release, repository, assets, public=args.public)
    report["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
