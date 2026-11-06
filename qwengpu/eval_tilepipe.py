"""Accumulate both preloaded K slices in one model-scheduled launch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import transformers

from qwengpu.eval_cached_tile import FIXTURE
from qwengpu.eval_tile8 import (
    RESIDENT_TILEPIPE_LAYOUT,
    load_tile_model,
    parse_launch,
)
from qwengpu.run_matmul16 import TILE, _tile, launch_two_slices, parse_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    meta = (
        json.loads((args.model / "qwengpu9b.json").read_text()) if args.model else None
    )
    if meta and meta["layout"] != RESIDENT_TILEPIPE_LAYOUT:
        parser.error("expected the tilepipe checkpoint")
    offset = meta["chip_token_offset"] if meta else 0
    model, circuit = load_tile_model(
        args.model, args.device, resident_add32=True, resident_tilepipe=True
    )
    payload = FIXTURE.read_bytes()
    a, b, initial = parse_inputs(json.loads(payload))
    first = parse_launch(
        {"a": _tile(a, 0, 0), "b": _tile(b, 0, 0), "c": _tile(initial, 0, 0)},
        looping=True,
        resident_add32=True,
    )
    alternate = _tile(a, 0, TILE) + _tile(b, TILE, 0)
    observed, audit, passes = launch_two_slices(
        model, circuit, first, alternate, args.device, offset
    )
    for index, step in enumerate(passes):
        scalar = [
            [
                (
                    initial[row][col]
                    + sum(a[row][k] * b[k][col] for k in range((index + 1) * TILE))
                )
                & 65535
                for col in range(TILE)
            ]
            for row in range(TILE)
        ]
        if step["matrix"] != scalar:
            raise ValueError(f"slice {index} differs from the scalar result")

    report = {
        "device": args.device,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "merge": meta,
        "input_sha256": hashlib.sha256(payload).hexdigest(),
        "passes": passes,
        "memory_tokens_by_call": audit,
        **observed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print("One launch, one model-generated transition, both scalar results exact.")


if __name__ == "__main__":
    main()
