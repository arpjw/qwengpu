"""Save checkpoint metadata and streaming weight hashes, without copying weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    names = (
        "qwengpu9b.json",
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
    )
    files = {name: json.loads((args.model / name).read_text()) for name in names}
    shards = set(files["model.safetensors.index.json"]["weight_map"].values())
    hashes = {}
    for name in sorted(shards | set(names)):
        with (args.model / name).open("rb") as stream:
            hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "checkpoint_source_commit": args.source,
        "compiler_commit": args.compiler,
        "base_revision": args.base,
        "weights_stored_in_git": False,
        "rebuild": f"python -m qwengpu.merge9b /path/to/pinned/Qwen3.5-9B --{args.variant} --out /path/to/rebuilt/{args.variant}",
        "files": files,
        "sha256": hashes,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved metadata and hashes for {len(shards)} weight shards")


if __name__ == "__main__":
    main()
