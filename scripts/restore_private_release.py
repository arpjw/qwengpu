"""Restore a checksum-pinned checkpoint from a private GitHub draft release.

Read a GitHub token from standard input. Download one asset at a time so the
restore needs space only for the checkpoint and one chunk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run_gh(args: list[str], token: str) -> str:
    result = subprocess.run(
        ["gh", *args], check=True, text=True, capture_output=True,
        env=dict(os.environ, GH_TOKEN=token),
    )
    return result.stdout


def download(repo: str, tag: str, name: str, directory: Path, token: str) -> Path:
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError(f"unsafe asset name: {name!r}")
    run_gh(["release", "download", tag, "--pattern", name,
            "--dir", str(directory), "-R", repo], token)
    path = directory / name
    if not path.is_file():
        raise RuntimeError(f"release asset missing: {name}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo")
    parser.add_argument("tag")
    parser.add_argument("release_id", type=int)
    parser.add_argument("expected_checksum_digest")
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    token = sys.stdin.readline().strip()
    if not token:
        parser.error("provide a GitHub token on standard input")
    if args.out.exists():
        parser.error("restore destination must not exist")

    info = json.loads(
        run_gh(["api", f"repos/{args.repo}/releases/{args.release_id}"], token)
    )
    if not info["draft"] or info["tag_name"] != args.tag:
        raise RuntimeError("source must be the expected draft release")
    repo_info = json.loads(run_gh(["api", f"repos/{args.repo}"], token))
    if not repo_info["private"]:
        raise RuntimeError("source repository must be private")

    with tempfile.TemporaryDirectory(prefix="qwengpu-restore-") as scratch:
        temporary = Path(scratch)
        manifest_path = download(args.repo, args.tag, "github-backup-manifest.json",
                                 temporary, token)
        manifest = json.loads(manifest_path.read_text())
        if any(manifest[key] != value for key, value in (
            ("repo", args.repo), ("tag", args.tag),
            ("release_id", args.release_id),
            ("package_checksum_digest", args.expected_checksum_digest),
        )):
            raise ValueError("backup manifest identity differs")
        manifest_path.unlink()
        checksum = download(args.repo, args.tag, "SHA256SUMS", temporary, token)
        if digest(checksum) != args.expected_checksum_digest:
            raise ValueError("downloaded checkpoint checksum manifest differs")
        expected = {
            name: sha for sha, name in (
                line.split("  ", 1) for line in checksum.read_text().splitlines()
            )
        }
        checksum.unlink()

        grouped: dict[str, list[dict]] = defaultdict(list)
        for part in manifest["parts"]:
            name = part["file"]
            if Path(name).name != name or name in {"", ".", ".."}:
                raise ValueError(f"unsafe checkpoint name: {name!r}")
            grouped[name].append(part)
        if set(grouped) != set(expected) | {"SHA256SUMS"}:
            raise ValueError("backup and checkpoint file lists differ")

        args.out.mkdir(parents=True)
        for name, parts in grouped.items():
            target = args.out / name
            file_hash = hashlib.sha256()
            offset = 0
            with target.open("xb") as output:
                for part in parts:
                    if part["offset"] != offset:
                        raise ValueError(f"non-contiguous backup parts: {name}")
                    asset = download(args.repo, args.tag, part["asset"], temporary, token)
                    if asset.stat().st_size != part["size"] or digest(asset) != part["sha256"]:
                        raise ValueError(f"backup part differs: {part['asset']}")
                    with asset.open("rb") as stream:
                        while block := stream.read(8 * 1024 * 1024):
                            output.write(block)
                            file_hash.update(block)
                    asset.unlink()
                    offset += part["size"]
            wanted = (args.expected_checksum_digest if name == "SHA256SUMS"
                      else expected[name])
            if file_hash.hexdigest() != wanted:
                raise ValueError(f"restored file differs: {name}")
            print(f"{name}: restored and verified ({offset} bytes)", flush=True)
    print(f"Verified {len(grouped)} restored files at {args.out}")


if __name__ == "__main__":
    main()
