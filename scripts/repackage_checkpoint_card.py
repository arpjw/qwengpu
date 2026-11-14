"""Replace only a restored checkpoint's model card and recheck its package."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from check_checkpoint_package import CHECKSUMS, package_hashes, sha256


def checksum_rows(contents: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in contents.splitlines():
        digest, name = line.split("  ", 1)
        if name in rows or len(digest) != 64:
            raise ValueError("invalid or duplicate checksum entry")
        rows[name] = digest
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("card", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pinned-manifest", type=Path, required=True)
    parser.add_argument("--expected-original-checksum-digest", required=True)
    parser.add_argument("--receipt-out", type=Path)
    args = parser.parse_args()

    if args.output.exists():
        parser.error("output directory already exists")
    if not args.original.is_dir() or not args.card.is_file():
        parser.error("original package or replacement card is missing")
    original_checksums = args.original / CHECKSUMS
    if sha256(original_checksums) != args.expected_original_checksum_digest:
        parser.error("original checksum manifest digest differs")
    old_rows = checksum_rows(original_checksums.read_text(encoding="utf-8"))
    original_files = {
        path.name: path for path in args.original.iterdir()
        if path.name != CHECKSUMS
    }
    if set(original_files) != set(old_rows):
        parser.error("original package file list differs from its checksums")
    if any(not path.is_file() or path.is_symlink() for path in original_files.values()):
        parser.error("original package contains a non-regular file")

    with tempfile.TemporaryDirectory(
        prefix=".qwengpu-card-", dir=args.output.parent
    ) as temporary:
        working = Path(temporary)
        for name, source in original_files.items():
            if name == "README.md":
                (working / name).write_bytes(args.card.read_bytes())
            else:
                os.link(source, working / name)
        contents = package_hashes(working, args.pinned_manifest)
        new_rows = checksum_rows(contents)
        changed = {name for name in old_rows if old_rows[name] != new_rows.get(name)}
        if set(old_rows) != set(new_rows) or changed != {"README.md"}:
            raise ValueError(f"expected only README.md to change, got {sorted(changed)}")
        (working / CHECKSUMS).write_text(contents, encoding="utf-8")
        receipt = {
            "original_checksum_digest": args.expected_original_checksum_digest,
            "corrected_checksum_digest": sha256(working / CHECKSUMS),
            "original_card_sha256": old_rows["README.md"],
            "corrected_card_sha256": new_rows["README.md"],
            "unchanged_files": len(old_rows) - 1,
            "package_files": len(old_rows),
        }
        working.rename(args.output)

    if args.receipt_out is not None:
        args.receipt_out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
