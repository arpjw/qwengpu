"""Verify a model-generated workgroup boundary in the two-warp tile circuit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import transformers

from qwengpu.compiled_tile8 import (
    C_MEMORY,
    NEXT_BANK,
    NEXT_WORKGROUP,
    SECOND_OUTPUT,
    TICKET,
    build_standalone,
)
from qwengpu.eval_cached_tile import FIXTURE
from qwengpu.eval_tile8 import launch_cached_tile, oracle, parse_launch
from qwengpu.run_matmul16 import TILE, _tile, _trace_digest, parse_inputs


def qualify(
    model,
    circuit,
    case: dict,
    alternate: list[int],
    device: str,
    *,
    second_output: list[int] | None = None,
    right_operands: tuple[list[int], list[int]] | None = None,
) -> dict:
    _, observed, audit = launch_cached_tile(
        model,
        circuit,
        case,
        device,
        0,
        alternate_operands=alternate,
        auto_operand_bank=True,
        auto_workgroups=True,
        second_output=second_output,
        right_operands=right_operands,
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
        or (
            second_output is not None
            and observed["workgroup_outputs"][0] != group["matrix"]
        )
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
    current_alternate = alternate
    for index, end in enumerate([*boundaries, None]):
        if index == 2 and second_output is not None:
            reference_case["c"] = second_output
            if right_operands is not None:
                reference_case["b"] = right_operands[0]
                current_alternate = alternate[:64] + right_operands[1]
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
            alternate_operands=current_alternate,
            operand_bank=bank,
        )
        if index >= 2 and second_output is not None:
            for step in reference["trace"]:
                step["patch"] = [
                    token + SECOND_OUTPUT - C_MEMORY
                    if C_MEMORY <= token < C_MEMORY + 4096
                    else token
                    for token in step["patch"]
                ]
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
        **(
            {"workgroup_outputs": observed["workgroup_outputs"]}
            if second_output is not None
            else {}
        ),
        "exact": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--row-pairs", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--separate-outputs", action="store_true")
    parser.add_argument(
        "--right-panel",
        action="store_true",
        help="select different B and C tiles for the second workgroup",
    )
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
    right = (_tile(b, 0, TILE), _tile(b, TILE, TILE)) if args.right_panel else None
    model, circuit = build_standalone(resident_tilegrid=True)
    model.to(args.device).eval()
    result = qualify(
        model,
        circuit,
        case,
        alternate,
        args.device,
        second_output=(
            _tile(initial, 0, TILE)
            if args.separate_outputs or args.right_panel
            else None
        ),
        right_operands=right,
    )
    if args.right_panel:
        left, right = result["workgroup_outputs"]
        matrix = [l + r for l, r in zip(left, right, strict=True)]
        expected = [
            [
                (initial[i][j] + sum(a[i][k] * b[k][j] for k in range(2 * TILE)))
                & 65535
                if i < 2 * args.row_pairs
                else initial[i][j]
                for j in range(2 * TILE)
            ]
            for i in range(TILE)
        ]
        if matrix != expected:
            raise ValueError("resident panel multiply differs from the scalar matrix")
        result["matrix"] = matrix
    report = {
        "device": args.device,
        "row_pairs": args.row_pairs,
        "separate_outputs": args.separate_outputs or args.right_panel,
        "right_panel": args.right_panel,
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
