"""Back up a flat checkpoint as verified assets on a private draft release.

Read a GitHub token from standard input. Large files are split into assets
below GitHub's 2 GiB per-asset limit. The release remains a draft.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


CHUNK_BYTES = 1_700_000_000
CHECKSUMS = "SHA256SUMS"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run_gh(args: list[str], token: str) -> str:
    env = dict(os.environ, GH_TOKEN=token)
    result = subprocess.run(
        ["gh", *args], check=True, text=True, capture_output=True, env=env
    )
    return result.stdout


def assets(repo: str, release_id: int, token: str) -> dict[str, dict]:
    raw = run_gh(
        ["api", f"repos/{repo}/releases/{release_id}/assets?per_page=100"], token
    )
    entries = json.loads(raw)
    if len(entries) >= 100:
        raise RuntimeError("release asset listing needs pagination")
    return {entry["name"]: entry for entry in entries}


def upload(
    path: Path, name: str, sha256: str, size: int, repo: str,
    tag: str, release_id: int, token: str,
) -> None:
    existing = assets(repo, release_id, token).get(name)
    if existing is not None:
        if existing.get("size") != size or existing.get("digest") != f"sha256:{sha256}":
            raise RuntimeError(f"existing release asset differs: {name}")
        print(f"{name}: verified existing asset", flush=True)
        return
    if path.name != name:
        raise ValueError("upload path and asset name differ")
    run_gh(["release", "upload", tag, str(path), "-R", repo], token)
    saved = assets(repo, release_id, token).get(name)
    if saved is None or saved.get("size") != size or saved.get("digest") != f"sha256:{sha256}":
        raise RuntimeError(f"uploaded release asset did not verify: {name}")
    print(f"{name}: uploaded and verified ({size} bytes)", flush=True)


def package_files(package: Path, expected_digest: str) -> list[tuple[str, str]]:
    checksums = package / CHECKSUMS
    if digest(checksums) != expected_digest:
        raise ValueError("checkpoint checksum manifest digest differs")
    rows = [tuple(line.split("  ", 1)) for line in checksums.read_text().splitlines()]
    names = {name for _, name in rows}
    if names != {path.name for path in package.iterdir()} - {CHECKSUMS}:
        raise ValueError("checkpoint package file list differs from SHA256SUMS")
    if any(not (package / name).is_file() or (package / name).is_symlink() for name in names):
        raise ValueError("package contains a non-regular file")
    return [(CHECKSUMS, expected_digest), *[(name, sha) for sha, name in rows]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("repo")
    parser.add_argument("tag")
    parser.add_argument("release_id", type=int)
    parser.add_argument("expected_checksum_digest")
    parser.add_argument("--manifest-out", type=Path, required=True)
    args = parser.parse_args()
    token = sys.stdin.readline().strip()
    if not token:
        parser.error("provide a GitHub token on standard input")
    info = json.loads(
        run_gh(["api", f"repos/{args.repo}/releases/{args.release_id}"], token)
    )
    if not info["draft"] or info["tag_name"] != args.tag:
        raise RuntimeError("destination must be the expected draft release")
    repo_info = json.loads(run_gh(["api", f"repos/{args.repo}"], token))
    if not repo_info["private"]:
        raise RuntimeError("destination repository must be private")

    entries = package_files(args.package, args.expected_checksum_digest)
    records = []
    with tempfile.TemporaryDirectory(prefix="qwengpu-release-") as scratch:
        temporary = Path(scratch)
        for filename, expected_sha in entries:
            source = args.package / filename
            size = source.stat().st_size
            if size < CHUNK_BYTES:
                actual_sha = digest(source)
                if actual_sha != expected_sha:
                    raise ValueError(f"qualified file changed: {filename}")
                upload(source, filename, actual_sha, size, args.repo, args.tag,
                       args.release_id, token)
                records.append({"file": filename, "asset": filename, "offset": 0,
                                "size": size, "sha256": actual_sha})
                continue

            full_hash = hashlib.sha256()
            with source.open("rb") as stream:
                part = 0
                offset = 0
                while offset < size:
                    name = f"{filename}.part{part:03d}"
                    chunk = temporary / name
                    part_hash = hashlib.sha256()
                    remaining = min(CHUNK_BYTES, size - offset)
                    with chunk.open("wb") as output:
                        while remaining:
                            block = stream.read(min(8 * 1024 * 1024, remaining))
                            if not block:
                                raise IOError(f"short read: {filename}")
                            output.write(block)
                            full_hash.update(block)
                            part_hash.update(block)
                            remaining -= len(block)
                    part_size = chunk.stat().st_size
                    upload(chunk, name, part_hash.hexdigest(), part_size, args.repo,
                           args.tag, args.release_id, token)
                    records.append({"file": filename, "asset": name, "offset": offset,
                                    "size": part_size, "sha256": part_hash.hexdigest()})
                    chunk.unlink()
                    offset += part_size
                    part += 1
            if full_hash.hexdigest() != expected_sha:
                raise ValueError(f"qualified file changed: {filename}")

        manifest = {
            "repo": args.repo,
            "tag": args.tag,
            "release_id": args.release_id,
            "package_checksum_digest": args.expected_checksum_digest,
            "parts": records,
        }
        args.manifest_out.write_text(json.dumps(manifest, indent=2) + "\n")
        upload(args.manifest_out, args.manifest_out.name, digest(args.manifest_out),
               args.manifest_out.stat().st_size, args.repo, args.tag,
               args.release_id, token)
    print(f"Verified {len(entries)} packaged files across {len(records)} release assets")


if __name__ == "__main__":
    main()
