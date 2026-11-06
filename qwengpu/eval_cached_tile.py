"""Accumulate two K slices into an 8×8 tile in one inference cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import transformers

from qwengpu.eval_tile8 import (
    RESIDENT_ADD32_LAYOUTS,
    RESIDENT_TILEBANKS_LAYOUT,
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


def launch_slice(
    model,
    circuit,
    case,
    device,
    offset,
    *,
    cache=None,
    alternate_operands=None,
    operand_bank=0,
) -> tuple:
    cache, result, audit = launch_cached_tile(
        model,
        circuit,
        case,
        device,
        offset,
        cache=cache,
        alternate_operands=alternate_operands,
        operand_bank=operand_bank,
    )
    transport = result.pop("transport")
    expected = oracle(
        case,
        256,
        looping=True,
        alternate_operands=alternate_operands,
        operand_bank=operand_bank,
    )
    if result != expected or result["outcome"] != "halted":
        raise ValueError("tile slice differs from the interpreter")
    if not all(
        transport[f"resident_{bank}_exact"]
        for bank in ("grid", "memory", "rom", "state")
    ):
        raise ValueError("tile cache rows differ from the expected state")
    if alternate_operands is not None and not transport["resident_operand_banks_exact"]:
        raise ValueError(
            "operand cache rows or the bank selector differ from the input"
        )
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
    parser.add_argument("--preload-operands", action="store_true")
    parser.add_argument("--device", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    meta = (
        json.loads((args.model / "qwengpu9b.json").read_text()) if args.model else None
    )
    if meta and meta["layout"] not in RESIDENT_ADD32_LAYOUTS:
        parser.error("expected a resident ADD32 checkpoint")
    if args.preload_operands and meta and meta["layout"] != RESIDENT_TILEBANKS_LAYOUT:
        parser.error("--preload-operands requires the tilebanks checkpoint")
    offset = meta["chip_token_offset"] if meta else 0
    model, circuit = load_tile_model(
        args.model,
        args.device,
        resident_add32=True,
        resident_tilebanks=args.preload_operands,
    )
    payload = FIXTURE.read_bytes()
    a, b, initial = parse_inputs(json.loads(payload))
    c = _tile(initial, 0, 0)
    cache = None
    passes = []
    alternate = _tile(a, 0, TILE) + _tile(b, TILE, 0) if args.preload_operands else None
    for k in (0, TILE):
        initial_c = c
        source_k = 0 if args.preload_operands else k
        case = parse_launch(
            {"a": _tile(a, 0, source_k), "b": _tile(b, source_k, 0), "c": initial_c},
            looping=True,
            resident_add32=True,
        )
        cache, step = launch_slice(
            model,
            circuit,
            case,
            args.device,
            offset,
            cache=cache,
            alternate_operands=alternate,
            operand_bank=k // TILE if args.preload_operands else 0,
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
    cold_case = parse_launch(
        {"a": _tile(a, 0, TILE), "b": _tile(b, TILE, 0), "c": initial_c},
        looping=True,
        resident_add32=True,
    )
    _, cold = launch_slice(model, circuit, cold_case, args.device, offset)
    saved_tokens = 512 if args.preload_operands else 256
    if (
        cold["trace_sha256"] != passes[-1]["trace_sha256"]
        or cold["matrix"] != passes[-1]["matrix"]
        or cold["transport"]["input_tokens"] - passes[-1]["transport"]["input_tokens"]
        != saved_tokens
    ):
        raise ValueError("retained second slice differs from its fresh-cache control")
    report = {
        "device": args.device,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "merge": meta,
        "input_sha256": hashlib.sha256(payload).hexdigest(),
        "passes": passes,
        **({"preloaded_operands": True} if args.preload_operands else {}),
        "cold_second_slice": cold,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Both slices exact. Second slice reused C and saved {saved_tokens} upload tokens."
    )


if __name__ == "__main__":
    main()
