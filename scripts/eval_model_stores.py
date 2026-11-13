"""Verify fixed lane commit tokens replace indexed global RAM writeback tags."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwengpu.eval_globalgrid import launch
from qwengpu.eval_tile8 import load_tile_model

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device", default="cuda", choices=("cuda", "mps"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model, circuit = load_tile_model(args.model, args.device, internal_stores=True)
    meta = (
        json.loads((args.model / "qwengpu9b.json").read_bytes()) if args.model else None
    )
    offset = meta["chip_token_offset"] if meta else 0
    cases = []
    for folder, name in (
        ("resident_alu32_v1", "address_walk"),
        ("resident_interleaved_v1", "interleaved_publish"),
        ("scoped_memory_v1", "isolation"),
    ):
        data = json.loads((ROOT / "eval/frozen" / folder / f"{name}.json").read_bytes())
        _, observed = launch(model, circuit, data, args.device, offset)
        transport = observed["transport"]
        assert transport["internal_store_words"] > 0
        assert transport["matrix_patch_tokens"] == 0
        assert sum(call["memory"] for call in observed["audit"]) == 1024
        assert transport["internal_store_read_tags"] == 4 * transport["internal_store_words"]
        assert sum(len(call["commits"]) for call in observed["audit"]) == transport["internal_store_words"]
        if "scoped_words" in observed:
            assert transport["internal_scoped_words"] > 0 and transport["scoped_patch_tokens"] == 0
            assert sum(call["scoped"] for call in observed["audit"]) == 512
        cases.append({"name": name, "result": observed})
    args.out.write_text(json.dumps({"device": args.device, "merge": meta, "cases": cases}, indent=2) + "\n")


if __name__ == "__main__":
    main()
