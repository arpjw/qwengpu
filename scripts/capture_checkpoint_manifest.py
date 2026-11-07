"""Save checkpoint metadata and streaming weight hashes, without copying weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
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
    dtype_bytes: dict[str, int] = {}
    non_bf16 = []
    for name in sorted(shards):
        with (args.model / name).open("rb") as stream:
            header = json.loads(stream.read(struct.unpack("<Q", stream.read(8))[0]))
        for tensor, item in header.items():
            if tensor == "__metadata__":
                continue
            size = item["data_offsets"][1] - item["data_offsets"][0]
            dtype = item["dtype"]
            dtype_bytes[dtype] = dtype_bytes.get(dtype, 0) + size
            if dtype != "BF16":
                non_bf16.append(
                    {
                        "name": tensor,
                        "dtype": dtype,
                        "shape": item["shape"],
                        "bytes": size,
                    }
                )
    assert (
        sum(dtype_bytes.values())
        == files["model.safetensors.index.json"]["metadata"]["total_size"]
    )
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
        "dtype_bytes": dtype_bytes,
        "non_bf16": non_bf16,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved metadata and hashes for {len(shards)} weight shards")


if __name__ == "__main__":
    main()
