"""Accumulate both preloaded K slices in one model-scheduled launch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import transformers

from qwengpu.compiled_tile8 import NEXT_BANK, TICKET
from qwengpu.eval_cached_tile import FIXTURE
from qwengpu.eval_tile8 import (
    RESIDENT_TILEPIPE_LAYOUT,
    launch_cached_tile,
    load_tile_model,
    oracle,
    parse_launch,
)
from qwengpu.run_matmul16 import TILE, _tile, _trace_digest, parse_inputs


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
    _, observed, audit = launch_cached_tile(
        model,
        circuit,
        first,
        args.device,
        offset,
        alternate_operands=alternate,
        auto_operand_bank=True,
    )
    transition = observed["bank_transition"]
    transport = observed["transport"]
    if (
        observed["outcome"] != "halted"
        or not transition
        or transition["ticket"] != NEXT_BANK
        or not transition["resident_memory_exact"]
        or transition["bank_before"] != 0
        or not all(
            transport[f"resident_{name}_exact"]
            for name in ("grid", "memory", "rom", "state", "operand_banks")
        )
        or sum(call["grid"] for call in audit) != 1
        or sum(call["transition"] for call in audit) != 1
    ):
        raise ValueError("model-scheduled transition or physical cache check failed")
    first_trace = observed["trace"][: transition["instructions"]]
    second_trace = observed["trace"][transition["instructions"] :]
    first_schedule = observed["schedules"][: transition["schedule_index"]]
    second_schedule = observed["schedules"][transition["schedule_index"] :]
    second_schedule[0] = {**second_schedule[0], "ticket": TICKET}
    passes = []
    c = first["c"]
    for index, (trace, schedule) in enumerate(
        ((first_trace, first_schedule), (second_trace, second_schedule))
    ):
        reference = oracle(
            {**first, "c": c},
            256,
            looping=True,
            alternate_operands=alternate,
            operand_bank=index,
        )
        # The first slice emits a transition in place of its separate HALT ticket.
        if index == 0:
            schedule.append(reference["schedules"][-1])
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
        matrix = transition["matrix"] if index == 0 else observed["matrix"]
        if trace != reference["trace"] or schedule != reference["schedules"]:
            raise ValueError(f"slice {index} differs from the separate execution trace")
        if matrix != reference["matrix"] or matrix != scalar:
            raise ValueError(f"slice {index} differs from the scalar result")
        passes.append(
            {
                "k": index * TILE,
                "matrix": matrix,
                "instructions": len(trace),
                "scheduler_decisions": len(schedule),
                "trace_sha256": _trace_digest({"trace": trace, "schedules": schedule}),
            }
        )
        c = [word for row in scalar for word in row]

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
