"""Verify a model-generated workgroup boundary in the two-warp tile circuit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import transformers

from qwengpu.compiled_tile8 import NEXT_BANK, NEXT_WORKGROUP, TICKET, build_standalone
from qwengpu.eval_cached_tile import FIXTURE
from qwengpu.eval_tile8 import launch_cached_tile, oracle, parse_launch
from qwengpu.run_matmul16 import TILE, _tile, _trace_digest, parse_inputs


def qualify(model, circuit, case: dict, alternate: list[int], device: str) -> dict:
    _, observed, audit = launch_cached_tile(
        model,
        circuit,
        case,
        device,
        0,
        alternate_operands=alternate,
        auto_operand_bank=True,
        auto_workgroups=True,
    )
    banks = observed["bank_transitions"]
    group = observed["workgroup_transition"]
    transport = observed["transport"]
    if (
        len(banks) != 2
        or group is None
        or group["group_before"] != 0
        or group["bank_before"] != 1
        or observed["workgroup_cursor"] != 1
        or [item["bank_before"] for item in banks] != [0, 0]
        or observed["outcome"] != "halted"
        or not all(
            transport[f"resident_{name}_exact"]
            for name in ("grid", "memory", "rom", "state", "operand_banks")
        )
        or sum(call["transition"] for call in audit) != 2
        or sum(call["workgroup_transition"] for call in audit) != 1
        or sum(call["grid"] for call in audit) != 1
    ):
        raise ValueError("workgroup or bank transition audit failed")
    boundaries = [banks[0], group, banks[1]]
    if not all(item["resident_memory_exact"] for item in boundaries):
        raise ValueError("resident memory differed at a workgroup or bank boundary")
    if [item["ticket"] for item in boundaries] != [
        NEXT_BANK,
        NEXT_WORKGROUP,
        NEXT_BANK,
    ]:
        raise ValueError("unexpected transition sequence")

    instruction_index = schedule_index = 0
    results = []
    reference_case = case.copy()
    for index, end in enumerate([*boundaries, None]):
        trace = observed["trace"][
            instruction_index : end["instructions"] if end else None
        ]
        schedules = observed["schedules"][
            schedule_index : end["schedule_index"] if end else None
        ]
        bank = index % 2
        reference = oracle(
            reference_case,
            256,
            looping=True,
            alternate_operands=alternate,
            operand_bank=bank,
        )
        if index:
            schedules[0] = {**schedules[0], "ticket": TICKET}
        if end:
            schedules.append(reference["schedules"][-1])
        matrix = end["matrix"] if end else observed["matrix"]
        if (trace, schedules, matrix) != (
            reference["trace"],
            reference["schedules"],
            reference["matrix"],
        ):
            raise ValueError(
                f"workgroup {index // 2} operand bank {bank} differs from reference"
            )
        results.append(
            {
                "workgroup": index // 2,
                "bank": bank,
                "matrix": matrix,
                "instructions": len(trace),
                "trace_sha256": _trace_digest({"trace": trace, "schedules": schedules}),
            }
        )
        reference_case.update(
            c=[value for row in matrix for value in row],
            initial_registers=reference["warp_registers"],
        )
        if end:
            instruction_index = end["instructions"]
            schedule_index = end["schedule_index"]
    return {
        "passes": results,
        "transitions": boundaries,
        "transport": transport,
        "memory_tokens_by_call": audit,
        "instructions": len(observed["trace"]),
        "scheduler_decisions": len(observed["schedules"]),
        "exact": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--row-pairs", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = FIXTURE.read_bytes()
    a, b, initial = parse_inputs(json.loads(payload))
    case = parse_launch(
        {
            "a": _tile(a, 0, 0),
            "b": _tile(b, 0, 0),
            "c": _tile(initial, 0, 0),
            "row_pairs": args.row_pairs,
        },
        resident_loop32=True,
    )
    alternate = _tile(a, 0, TILE) + _tile(b, TILE, 0)
    model, circuit = build_standalone(resident_tilegrid=True)
    model.to(args.device).eval()
    result = qualify(model, circuit, case, alternate, args.device)
    report = {
        "device": args.device,
        "row_pairs": args.row_pairs,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "fixture_sha256": hashlib.sha256(payload).hexdigest(),
        "circuit": circuit.stats(),
        **result,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{result['instructions']} instructions exact on {args.device}")


if __name__ == "__main__":
    main()
