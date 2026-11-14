"""Restore a checksum-pinned checkpoint from a GitHub release.

Private draft mode reads a GitHub token from standard input. Public mode
uses the local gh login. Download one asset at a time so the restore needs
space only for the checkpoint and one chunk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run_gh(args: list[str], token: str) -> str:
    env = dict(os.environ)
    if token:
        env["GH_TOKEN"] = token
    result = subprocess.run(
        ["gh", *args], check=True, text=True, capture_output=True,
        env=env,
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


def resume_files(directory: Path, wanted: dict[str, str]) -> set[str]:
    """Verify completed files and remove only incomplete expected files."""
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("resume destination must be a real directory")
    entries = list(directory.iterdir())
    if any(entry.name not in wanted for entry in entries):
        raise ValueError("resume destination contains an unexpected file")
    for entry in entries:
        if not stat.S_ISREG(entry.lstat().st_mode):
            raise ValueError(f"resume destination contains a non-file: {entry.name}")
    completed: set[str] = set()
    for entry in entries:
        if digest(entry) == wanted[entry.name]:
            completed.add(entry.name)
        else:
            entry.unlink()
    return completed


def check_source(info: dict, repo_info: dict, tag: str, public: bool) -> None:
    if info["tag_name"] != tag:
        raise RuntimeError("source release tag differs")
    if public:
        if info["draft"] or repo_info["private"]:
            raise RuntimeError("public mode requires a published release in a public repository")
    elif not info["draft"] or not repo_info["private"]:
        raise RuntimeError("private mode requires a draft release in a private repository")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo")
    parser.add_argument("tag")
    parser.add_argument("release_id", type=int)
    parser.add_argument("expected_checksum_digest")
    parser.add_argument("out", type=Path)
    parser.add_argument("--backup-manifest", default="github-backup-manifest.json")
    parser.add_argument("--public", action="store_true",
                        help="require a published release in a public repository")
    parser.add_argument("--resume", action="store_true",
                        help="verify existing expected files and resume an interrupted restore")
    args = parser.parse_args()
    token = "" if args.public else sys.stdin.readline().strip()
    if not args.public and not token:
        parser.error("provide a GitHub token on standard input")
    if args.out.exists() and not args.resume:
        parser.error("restore destination must not exist")
    if args.resume and not args.out.exists():
        parser.error("resume destination must exist")

    info = json.loads(
        run_gh(["api", f"repos/{args.repo}/releases/{args.release_id}"], token)
    )
    repo_info = json.loads(run_gh(["api", f"repos/{args.repo}"], token))
    check_source(info, repo_info, args.tag, args.public)

    with tempfile.TemporaryDirectory(prefix="qwengpu-restore-") as scratch:
        temporary = Path(scratch)
        manifest_path = download(args.repo, args.tag, args.backup_manifest,
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

        wanted = {**expected, "SHA256SUMS": args.expected_checksum_digest}
        if args.resume:
            completed = resume_files(args.out, wanted)
            print(f"Verified {len(completed)} existing files for resume", flush=True)
        else:
            args.out.mkdir(parents=True)
            completed = set()
        for name, parts in grouped.items():
            target = args.out / name
            if name in completed:
                print(f"{name}: already verified", flush=True)
                continue
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
            if file_hash.hexdigest() != wanted[name]:
                raise ValueError(f"restored file differs: {name}")
            print(f"{name}: restored and verified ({offset} bytes)", flush=True)
    print(f"Verified {len(grouped)} restored files at {args.out}")


if __name__ == "__main__":
    main()
