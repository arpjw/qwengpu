"""Use one ROM image at runtime lengths 0, 19, and 64 on the same cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwengpu.assemble import assemble
from qwengpu.eval_globalgrid import launch
from qwengpu.eval_tile8 import load_tile_model

PROGRAM = """
.warp 0
GLOBAL_ID
SAVE_INDEX
MASK_EXTENT
ADD32I 1
STORE_INDEX
HALT
.warp 1
GLOBAL_ID
SAVE_INDEX
MASK_EXTENT
ADD32I 1
STORE_INDEX
HALT
"""
EXTENTS = (19, 0, 64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device", default="mps", choices=("mps", "cuda"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    meta = (
        json.loads((args.model / "qwengpu9b.json").read_bytes()) if args.model else None
    )
    model, circuit = load_tile_model(args.model, args.device, full_integer=True)
    offset = meta["chip_token_offset"] if meta else 0
    cache = None
    words = [0xA51C] * 256
    runs = []
    for extent in EXTENTS:
        data = assemble(
            PROGRAM, {"workgroups": 4, "extent": extent, "global_words": words},
            full_integer=True,
        )
        retained, result = launch(model, circuit, data, args.device, offset, cache)
        assert cache is None or retained is cache
        cache = retained
        expected = words[:]
        expected[:extent] = list(range(1, extent + 1))
        assert result["global_words"] == expected
        transport = result["transport"]
        if circuit.internal_stores:
            assert transport["matrix_patch_tokens"] == 0
            assert transport["internal_store_commands"] == extent
        else:
            assert transport["matrix_patch_tokens"] == extent * 4
        assert result["audit"][0]["memory"] == (1024 if not runs else 0)
        assert not runs or result["programs"] == runs[0]["result"]["programs"]
        words = result["global_words"]
        runs.append({"extent": extent, "result": result})
    args.out.write_text(
        json.dumps({"device": args.device, "merge": meta, "runs": runs, "exact": True}, indent=2) + "\n"
    )
    print("Identical ROM, lengths 19, 0 and 64, exact output and guard words")


if __name__ == "__main__":
    main()
