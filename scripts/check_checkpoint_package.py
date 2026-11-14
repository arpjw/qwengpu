"""Create or verify checksums for a private QwenGPU checkpoint package."""

import argparse
import hashlib
import json
from pathlib import Path

CHECKSUMS = "SHA256SUMS"
REQUIRED = {
    "LICENSE", "README.md", "config.json", "generation_config.json",
    "model.safetensors.index.json", "qwengpu9b.json", "tokenizer.json",
    "tokenizer_config.json", "build-manifest.json", "environment.txt",
}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def package_hashes(directory, pinned_manifest):
    paths = {}
    for path in directory.iterdir():
        if path.name in (".cache", CHECKSUMS):
            continue
        if not path.is_file() or path.is_symlink() or "\n" in path.name or "\r" in path.name:
            raise ValueError(f"unexpected package path {path.name!r}")
        paths[path.name] = path
    if missing := REQUIRED - paths.keys():
        raise ValueError(f"missing package files: {sorted(missing)}")
    index = json.loads(paths["model.safetensors.index.json"].read_text())
    shards = set(index["weight_map"].values())
    if not shards or shards != {name for name in paths if name.endswith(".safetensors")}:
        raise ValueError("the weight index and packaged shards differ")
    build = json.loads(paths["build-manifest.json"].read_text())
    if build != json.loads(pinned_manifest.read_text()):
        raise ValueError("the packaged build manifest differs from the qualified original")
    hashes = {name: sha256(path) for name, path in sorted(paths.items())}
    for name in shards | set(build["files"]):
        if hashes.get(name) != build["sha256"].get(name):
            raise ValueError(f"qualified digest differs for {name}")
    return "".join(f"{digest}  {name}\n" for name, digest in hashes.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--pinned-manifest", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--expected-checksum-digest")
    args = parser.parse_args()
    checksums = args.package / CHECKSUMS
    try:
        if not args.write and sha256(checksums) != args.expected_checksum_digest:
            raise ValueError("checksum manifest digest differs from the original")
        contents = package_hashes(args.package, args.pinned_manifest)
        if args.write:
            with checksums.open("x", encoding="utf-8") as stream:
                stream.write(contents)
        elif checksums.read_text(encoding="utf-8") != contents:
            raise ValueError("one or more packaged files differ from the checksums")
    except (OSError, KeyError, ValueError) as exc:
        parser.error(str(exc))
    print(sha256(checksums))


if __name__ == "__main__":
    main()
