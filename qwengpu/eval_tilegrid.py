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
    OUTPUT_BANKS,
    RESIDENT_COMPUTE_LAYOUT,
    RESIDENT_GLOBALGRID_LAYOUT,
    RESIDENT_TILEGRID_LAYOUT,
    TICKET,
    TILEGRID_VOCAB,
    WORKGROUP_TICKETS,
)
from qwengpu.eval_cached_tile import FIXTURE
from qwengpu.eval_tile8 import (
    BottomRow,
    launch_cached_tile,
    load_tile_model,
    oracle,
    parse_launch,
)
from qwengpu.memory_geometry import MEMORY512_LAYOUT, WIDE_MEMORY
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
    bottom: BottomRow | None = None,
    offset: int = 0,
) -> dict:
    _, observed, audit = launch_cached_tile(
        model,
        circuit,
        case,
        device,
        offset,
        alternate_operands=alternate,
        auto_operand_bank=True,
        auto_workgroups=True,
        second_output=second_output,
        right_operands=right_operands,
        bottom=bottom,
    )
    banks = observed["bank_transitions"]
    groups = observed["workgroup_transitions"]
    count = 4 if bottom is not None else 2
    transport = observed["transport"]
    if (
        len(banks) != count
        or [group["group_before"] for group in groups] != list(range(count - 1))
        or [group["bank_before"] for group in groups] != [1] * (count - 1)
        or observed["workgroup_cursor"] != count - 1
        or [item["bank_before"] for item in banks] != [0] * count
        or observed["outcome"] != "halted"
        or not all(
            transport[f"resident_{name}_exact"]
            for name in ("grid", "memory", "rom", "state", "operand_banks")
        )
        or sum(call["transition"] for call in audit) != count
        or sum(call["workgroup_transition"] for call in audit) != count - 1
        or sum(call["grid"] for call in audit) != 1
        or (
            second_output is not None
            and observed["workgroup_outputs"][:-1]
            != [group["matrix"] for group in groups]
        )
    ):
        raise ValueError("workgroup or bank transition audit failed")
    boundaries = sorted(banks + groups, key=lambda item: item["schedule_index"])
    if not all(item["resident_memory_exact"] for item in boundaries):
        raise ValueError("resident memory differed at a workgroup or bank boundary")
    expected_tickets = [NEXT_BANK]
    for ticket in WORKGROUP_TICKETS[: count - 1]:
        expected_tickets.extend([ticket, NEXT_BANK])
    if [item["ticket"] for item in boundaries] != expected_tickets:
        raise ValueError("unexpected transition sequence")

    instruction_index = schedule_index = 0
    results = []
    reference_case = case.copy()
    current_alternate = alternate
    output_seeds = [case["c"], second_output]
    if bottom is not None:
        output_seeds.extend(bottom.outputs)
    for index, end in enumerate([*boundaries, None]):
        group, bank = divmod(index, 2)
        if bank == 0 and second_output is not None:
            a_panels = (
                bottom.operands
                if bottom is not None and group >= 2
                else (case["a"], alternate[:64])
            )
            b_panels = (
                right_operands
                if right_operands is not None and group % 2
                else (case["b"], alternate[64:])
            )
            reference_case.update(c=output_seeds[group], a=a_panels[0], b=b_panels[0])
            current_alternate = a_panels[1] + b_panels[1]
        trace = observed["trace"][
            instruction_index : end["instructions"] if end else None
        ]
        schedules = observed["schedules"][
            schedule_index : end["schedule_index"] if end else None
        ]
        reference = oracle(
            reference_case,
            256,
            looping=True,
            alternate_operands=current_alternate,
            operand_bank=bank,
        )
        if group and second_output is not None:
            for step in reference["trace"]:
                step["patch"] = [
                    token + OUTPUT_BANKS[group] - C_MEMORY
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
                f"workgroup {group} operand bank {bank} differs from reference"
            )
        results.append(
            {
                "workgroup": group,
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


def execute_matrix(
    model,
    circuit,
    a: list[list[int]],
    b: list[list[int]],
    initial: list[list[int]],
    device: str,
    *,
    row_pairs: int = 4,
    separate_outputs: bool = True,
    right_panel: bool = True,
    full_grid: bool = True,
    offset: int = 0,
) -> dict:
    """Run and verify one resident matrix grid using caller-supplied operands."""
    right_panel = right_panel or full_grid
    case = parse_launch(
        {
            "a": _tile(a, 0, 0),
            "b": _tile(b, 0, 0),
            "c": _tile(initial, 0, 0),
            "row_pairs": row_pairs,
        },
        resident_loop32=True,
    )
    alternate = _tile(a, 0, TILE) + _tile(b, TILE, 0)
    right = (_tile(b, 0, TILE), _tile(b, TILE, TILE)) if right_panel else None
    bottom = (
        BottomRow(
            operands=(_tile(a, TILE, 0), _tile(a, TILE, TILE)),
            outputs=(_tile(initial, TILE, 0), _tile(initial, TILE, TILE)),
        )
        if full_grid
        else None
    )
    result = qualify(
        model,
        circuit,
        case,
        alternate,
        device,
        second_output=(
            _tile(initial, 0, TILE) if separate_outputs or right_panel else None
        ),
        right_operands=right,
        bottom=bottom,
        offset=offset,
    )
    if right_panel:
        outputs = result["workgroup_outputs"]
        matrix = [
            l + r
            for index in range(0, len(outputs), 2)
            for l, r in zip(outputs[index], outputs[index + 1], strict=True)
        ]
        expected = [
            [
                (initial[i][j] + sum(a[i][k] * b[k][j] for k in range(2 * TILE)))
                & 65535
                if i % TILE < 2 * row_pairs
                else initial[i][j]
                for j in range(2 * TILE)
            ]
            for i in range(len(matrix))
        ]
        if matrix != expected:
            raise ValueError("resident panel multiply differs from the scalar matrix")
        result["matrix"] = matrix
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--input", type=Path, default=FIXTURE)
    parser.add_argument("--row-pairs", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--separate-outputs", action="store_true")
    parser.add_argument(
        "--right-panel",
        action="store_true",
        help="select different B and C tiles for the second workgroup",
    )
    parser.add_argument("--grid-2x2", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = args.input.read_bytes()
    a, b, initial = parse_inputs(json.loads(payload))
    meta = (
        json.loads((args.model / "qwengpu9b.json").read_text()) if args.model else None
    )
    if meta is not None and (
        meta.get("layout")
        not in (
            RESIDENT_TILEGRID_LAYOUT,
            RESIDENT_GLOBALGRID_LAYOUT,
            RESIDENT_COMPUTE_LAYOUT,
            MEMORY512_LAYOUT,
        )
        or meta.get("chip_vocab")
        != (
            WIDE_MEMORY.vocab
            if meta.get("layout") == MEMORY512_LAYOUT
            else TILEGRID_VOCAB
        )
    ):
        parser.error("expected a merged tilegrid checkpoint")
    model, circuit = load_tile_model(
        args.model, args.device, resident_add32=True, resident_tilegrid=True
    )
    right_panel = args.right_panel or args.grid_2x2
    result = execute_matrix(
        model,
        circuit,
        a,
        b,
        initial,
        args.device,
        row_pairs=args.row_pairs,
        separate_outputs=args.separate_outputs,
        right_panel=right_panel,
        full_grid=args.grid_2x2,
        offset=meta["chip_token_offset"] if meta else 0,
    )
    report = {
        "device": args.device,
        "row_pairs": args.row_pairs,
        "separate_outputs": args.separate_outputs or right_panel,
        "right_panel": right_panel,
        "grid_2x2": args.grid_2x2,
        "merge": meta,
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
