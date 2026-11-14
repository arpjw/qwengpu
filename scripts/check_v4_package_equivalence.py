"""Prove that the corrected v4 package changed only its model card.

This compares pinned checksum lists. Run the package checker on clean
restores separately to establish that both lists match their files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def read_checksums(path: Path, expected_digest: str) -> dict[str, str]:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_digest:
        raise ValueError(f"checksum list digest differs: {path}")
    checksums = {}
    for line in data.decode("utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"invalid checksum line: {path}")
        if not name or Path(name).name != name or name in checksums:
            raise ValueError(f"invalid or duplicate package name: {name!r}")
        checksums[name] = digest
    return checksums


def compare(original: dict[str, str], corrected: dict[str, str]) -> dict:
    if original.keys() != corrected.keys():
        raise ValueError("package file lists differ")
    changed = sorted(name for name in original if original[name] != corrected[name])
    if changed != ["README.md"]:
        raise ValueError(f"expected only README.md to change, got {changed}")
    shards = sorted(name for name in original if name.endswith(".safetensors"))
    if len(shards) != 35 or len(original) != 48:
        raise ValueError("unexpected v4 package or shard count")
    return {
        "package_content_files": len(original),
        "unchanged_content_files": len(original) - len(changed),
        "unchanged_weight_shards": len(shards),
        "changed_files": changed,
        "original_card_sha256": original["README.md"],
        "corrected_card_sha256": corrected["README.md"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("corrected", type=Path)
    parser.add_argument("original_digest")
    parser.add_argument("corrected_digest")
    args = parser.parse_args()
    try:
        original = read_checksums(args.original, args.original_digest)
        corrected = read_checksums(args.corrected, args.corrected_digest)
        result = compare(original, corrected)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        **result,
        "original_checksum_digest": args.original_digest,
        "corrected_checksum_digest": args.corrected_digest,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
