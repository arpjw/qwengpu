"""Accumulate two K slices into an 8×8 tile in one inference cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import transformers

from qwengpu.eval_tile8 import (
    RESIDENT_ADD32_LAYOUT,
    launch_cached_tile,
    load_tile_model,
    oracle,
    parse_launch,
)
from qwengpu.run_matmul16 import TILE, _tile, _trace_digest, parse_inputs

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "eval/frozen/matmul16_gate_v1/dense_seeded.json"
)


def launch_slice(model, circuit, case, device, offset, *, cache=None) -> tuple:
    cache, result, audit = launch_cached_tile(
        model, circuit, case, device, offset, cache=cache
    )
    transport = result.pop("transport")
    if result != oracle(case, 256, looping=True) or result["outcome"] != "halted":
        raise ValueError("tile slice differs from the interpreter")
    if not all(
        transport[f"resident_{bank}_exact"]
        for bank in ("grid", "memory", "rom", "state")
    ):
        raise ValueError("tile cache rows differ from the expected state")
    return cache, {
        "matrix": result["matrix"],
        "instructions": len(result["trace"]),
        "scheduler_decisions": len(result["schedules"]),
        "trace_sha256": _trace_digest(
            {key: result[key] for key in ("schedules", "trace")}
        ),
        "transport": transport,
        "memory_tokens_by_call": audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    meta = (
        json.loads((args.model / "qwengpu9b.json").read_text()) if args.model else None
    )
    if meta and meta["layout"] != RESIDENT_ADD32_LAYOUT:
        parser.error("expected a resident ADD32 checkpoint")
    offset = meta["chip_token_offset"] if meta else 0
    model, circuit = load_tile_model(args.model, args.device, resident_add32=True)
    payload = FIXTURE.read_bytes()
    a, b, initial = parse_inputs(json.loads(payload))
    c = _tile(initial, 0, 0)
    cache = None
    passes = []
    for k in (0, TILE):
        case = parse_launch(
            {"a": _tile(a, 0, k), "b": _tile(b, k, 0), "c": c},
            looping=True,
            resident_add32=True,
        )
        cache, step = launch_slice(
            model, circuit, case, args.device, offset, cache=cache
        )
        scalar = [
            [
                (
                    initial[row][col]
                    + sum(a[row][i] * b[i][col] for i in range(k + TILE))
                )
                & 65535
                for col in range(TILE)
            ]
            for row in range(TILE)
        ]
        if step["matrix"] != scalar:
            raise ValueError(f"tile slice K={k} differs from the scalar reference")
        c = [word for row in scalar for word in row]
        passes.append({"k": k, **step})
    _, cold = launch_slice(model, circuit, case, args.device, offset)
    if (
        cold["trace_sha256"] != passes[-1]["trace_sha256"]
        or cold["matrix"] != passes[-1]["matrix"]
        or cold["transport"]["input_tokens"] - passes[-1]["transport"]["input_tokens"]
        != 256
    ):
        raise ValueError("retained second slice differs from its fresh-cache control")
    report = {
        "device": args.device,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "merge": meta,
        "input_sha256": hashlib.sha256(payload).hexdigest(),
        "passes": passes,
        "cold_second_slice": cold,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print("Both slices exact. Second slice reused C and saved 256 upload tokens.")


if __name__ == "__main__":
    main()
