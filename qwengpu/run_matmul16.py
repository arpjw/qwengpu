"""Compute C + A @ B modulo 65536 with resident output tiles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from qwengpu.compiled_tile8 import (
    NEXT_BANK,
    RESIDENT_COMPUTE_LAYOUT,
    RESIDENT_GLOBALGRID_LAYOUT,
    RESIDENT_TILEGRID_LAYOUT,
    TICKET,
    TILEGRID_VOCAB,
    VOCAB,
    WARP_DIRTY_OFFSET,
)
from qwengpu.eval_tile import _unsigned
from qwengpu.eval_tile8 import (
    RESIDENT_ADD32_LAYOUTS,
    RESIDENT_ADDMEM32_LAYOUTS,
    RESIDENT_BRANCH32_LAYOUTS,
    RESIDENT_GRID_LAYOUT,
    RESIDENT_LOOP32_LAYOUTS,
    RESIDENT_SHARED32_LAYOUTS,
    RESIDENT_TILEPIPE_LAYOUTS,
    RESIDENT_TRANSPORTS,
    execute,
    launch_cached_tile,
    load_tile_model,
    oracle,
    parse_launch,
)

SIZE = 16
TILE = 8
FORMAT = "qwengpu-matmul16-v1"


def parse_inputs(
    data: object,
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    if not isinstance(data, dict):
        raise TypeError("matrix input must be an object")

    def matrix(name: str, maximum: int) -> list[list[int]]:
        rows = data.get(
            name, [[0] * SIZE for _ in range(SIZE)] if name == "c" else None
        )
        if not isinstance(rows, list) or len(rows) != SIZE:
            raise ValueError(f"{name} must contain {SIZE} rows")
        return [
            _unsigned(row, SIZE, maximum, f"{name}[{i}]") for i, row in enumerate(rows)
        ]

    return matrix("a", 255), matrix("b", 255), matrix("c", 65535)


def _tile(matrix: list[list[int]], row: int, col: int) -> list[int]:
    return [
        word for line in matrix[row : row + TILE] for word in line[col : col + TILE]
    ]


def _trace_digest(trace: dict) -> str:
    return hashlib.sha256(
        json.dumps(trace, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def launch_two_slices(
    model,
    circuit,
    first: dict,
    alternate: list[int],
    device: str,
    offset: int,
) -> tuple[dict, list[dict[str, int]], list[dict]]:
    """Launch one output tile and verify the model's two execution slices."""
    _, observed, audit = launch_cached_tile(
        model,
        circuit,
        first,
        device,
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
    registers = first["initial_registers"]
    for index, (trace, schedule) in enumerate(
        ((first_trace, first_schedule), (second_trace, second_schedule))
    ):
        reference = oracle(
            {**first, "c": c, "initial_registers": registers},
            256,
            looping=True,
            alternate_operands=alternate,
            operand_bank=index,
        )
        # The first slice emits a transition in place of its separate HALT ticket.
        if index == 0:
            schedule.append(reference["schedules"][-1])
        matrix = transition["matrix"] if index == 0 else observed["matrix"]
        if trace != reference["trace"] or schedule != reference["schedules"]:
            differences = [
                (i, [key for key in actual if actual[key] != expected[key]])
                for i, (actual, expected) in enumerate(zip(trace, reference["trace"]))
                if actual != expected
            ]
            raise ValueError(
                f"slice {index} differs from the reference: {differences[:4]}"
            )
        if matrix != reference["matrix"]:
            raise ValueError(f"slice {index} differs from the reference matrix")
        separate = (
            oracle(
                {**first, "c": c},
                256,
                looping=True,
                alternate_operands=alternate,
                operand_bank=index,
            )
            if index
            else reference
        )
        differences = [
            {
                "instruction": i,
                "fields": sorted(key for key in actual if actual[key] != old[key]),
            }
            for i, (actual, old) in enumerate(
                zip(trace, separate["trace"], strict=True)
            )
            if actual != old
        ]
        passes.append(
            {
                "k": index * TILE,
                "matrix": matrix,
                "instructions": len(trace),
                "scheduler_decisions": len(schedule),
                "trace_sha256": _trace_digest({"trace": trace, "schedules": schedule}),
                "separate_trace_sha256": _trace_digest(
                    {"trace": separate["trace"], "schedules": separate["schedules"]}
                ),
                "register_carry_differences": differences,
            }
        )
        c = [word for row in matrix for word in row]
        registers = reference["warp_registers"]

    return observed, audit, passes


def run_scheduled_matmul(
    model,
    circuit,
    device: str,
    offset: int,
    a: list[list[int]],
    b: list[list[int]],
    c: list[list[int]],
) -> dict:
    passes = [{"k": k, "matrix": [row[:] for row in c]} for k in (0, TILE)]
    tiles = []
    for row in range(0, SIZE, TILE):
        for col in range(0, SIZE, TILE):
            first = parse_launch(
                {"a": _tile(a, row, 0), "b": _tile(b, 0, col), "c": _tile(c, row, col)},
                resident_add32=True,
            )
            alternate = _tile(a, row, TILE) + _tile(b, TILE, col)
            observed, audit, tile_passes = launch_two_slices(
                model, circuit, first, alternate, device, offset
            )
            for step, tile in zip(passes, tile_passes, strict=True):
                for i, output_row in enumerate(tile["matrix"]):
                    step["matrix"][row + i][col : col + TILE] = output_row
            tiles.append(
                {
                    "row": row,
                    "col": col,
                    "instructions": len(observed["trace"]),
                    "scheduler_decisions": len(observed["schedules"]),
                    "bank_transition": observed["bank_transition"],
                    "passes": tile_passes,
                    "transport": observed["transport"],
                    "memory_tokens_by_call": audit,
                }
            )
    return {"matrix": passes[-1]["matrix"], "passes": passes, "tiles": tiles}


def run_matmul(
    model,
    circuit,
    device: str,
    offset: int,
    a: list[list[int]],
    b: list[list[int]],
    c: list[list[int]],
    *,
    resident_add32: bool = False,
    retain_c: bool = False,
) -> dict:
    if retain_c and not resident_add32:
        raise ValueError("retaining C requires a resident ADD32 checkpoint")
    passes = {
        k: {"k": k, "matrix": [row[:] for row in c], "launches": []}
        for k in range(0, SIZE, TILE)
    }
    tiles = [(row, col) for row in range(0, SIZE, TILE) for col in range(0, SIZE, TILE)]
    if retain_c:
        order = [(k, row, col) for row, col in tiles for k in passes]
    else:
        order = [(k, row, col) for k in passes for row, col in tiles]
    cache = None
    for k, row, col in order:
        if k == 0:
            cache = None
        previous = c if k == 0 else passes[k - TILE]["matrix"]
        matrix = passes[k]["matrix"]
        launch = parse_launch(
            {
                "a": _tile(a, row, k),
                "b": _tile(b, k, col),
                "c": _tile(previous, row, col),
            },
            resident_grid=True,
            resident_add32=resident_add32,
        )
        if retain_c:
            cache, result, audit = launch_cached_tile(
                model, circuit, launch, device, offset, cache=cache
            )
        else:
            result = execute(
                model,
                circuit,
                launch,
                device,
                256,
                offset,
                resident_grid=True,
                resident_add32=resident_add32,
                **RESIDENT_TRANSPORTS["fixed"],
            )
        transport = result["transport"]
        if (
            result["outcome"] != "halted"
            or transport["fresh_caches"] != int(not retain_c or k == 0)
            or not all(
                transport[f"resident_{bank}_exact"]
                for bank in ("grid", "memory", "rom", "state")
            )
        ):
            raise RuntimeError(
                f"tile ({row}, {col}), K={k} failed: {result['outcome']}, {transport}"
            )
        for i, output_row in enumerate(result["matrix"]):
            matrix[row + i][col : col + TILE] = output_row
        trace = {name: result[name] for name in ("trace", "schedules")}
        record = {
            "row": row,
            "col": col,
            "instructions": len(trace["trace"]),
            "scheduler_decisions": len(trace["schedules"]),
            "trace_sha256": _trace_digest(trace),
            "transport": transport,
        }
        if resident_add32:
            grid_digits = WARP_DIRTY_OFFSET // 4
            grid_trace = [
                {**step, "snapshot": step["snapshot"][-grid_digits:]}
                for step in trace["trace"]
            ]
            record["grid_trace_sha256"] = _trace_digest({**trace, "trace": grid_trace})
        if retain_c:
            record["memory_tokens_by_call"] = audit
        passes[k]["launches"].append(record)
    return {"matrix": passes[SIZE - TILE]["matrix"], "passes": list(passes.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--model", type=Path, help="merged resident grid or ADD32 checkpoint"
    )
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument(
        "--resident-compute",
        action="store_true",
        help="use the same standalone circuit as the common compute runner",
    )
    variants.add_argument(
        "--resident-add32",
        action="store_true",
        help="use the standalone checkpoint that also runs ADD32I",
    )
    variants.add_argument(
        "--resident-loop32",
        action="store_true",
        help="enable WHILE_BIT across all 32 register bits",
    )
    variants.add_argument(
        "--resident-branch32",
        action="store_true",
        help="enable IF_BIT across all 32 register bits",
    )
    variants.add_argument(
        "--resident-addmem32",
        action="store_true",
        help="enable 32-bit addition from two indexed shared-memory pairs",
    )
    variants.add_argument(
        "--resident-shared32",
        action="store_true",
        help="use the standalone checkpoint with indexed high-limb memory",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--retain-c", action="store_true", help="keep each output tile in one cache"
    )
    modes.add_argument(
        "--schedule-k",
        action="store_true",
        help="schedule both slices in one tilepipe launch",
    )
    modes.add_argument(
        "--grid-2x2",
        action="store_true",
        help="schedule all four output tiles in one tilegrid cache",
    )
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument(
        "--out", type=Path, help="write JSON to a file instead of stdout"
    )
    args = parser.parse_args()

    meta = None
    offset = 0
    resident_compute = args.resident_compute
    resident_tilegrid = args.grid_2x2 or resident_compute
    resident_tilepipe = args.schedule_k or resident_tilegrid
    resident_loop32 = args.resident_loop32 or resident_tilepipe
    resident_branch32 = args.resident_branch32 or resident_loop32
    resident_addmem32 = args.resident_addmem32 or resident_branch32
    resident_shared32 = args.resident_shared32 or resident_addmem32
    resident_add32 = args.resident_add32 or resident_shared32
    try:
        payload = args.input.read_bytes()
        a, b, c = parse_inputs(json.loads(payload))
        if (
            (args.schedule_k or args.grid_2x2)
            and args.model is None
            and any(
                (
                    args.resident_add32,
                    args.resident_shared32,
                    args.resident_addmem32,
                    args.resident_branch32,
                    args.resident_loop32,
                )
            )
        ):
            raise ValueError("scheduled modes select their own standalone checkpoint")
        if args.model:
            meta = json.loads((args.model / "qwengpu9b.json").read_text())
            if (
                not isinstance(meta, dict)
                or meta.get("layout")
                not in (RESIDENT_GRID_LAYOUT, *RESIDENT_ADD32_LAYOUTS)
                or meta.get("chip_vocab")
                != (
                    TILEGRID_VOCAB
                    if meta.get("layout")
                    in (
                        RESIDENT_TILEGRID_LAYOUT,
                        RESIDENT_GLOBALGRID_LAYOUT,
                        RESIDENT_COMPUTE_LAYOUT,
                    )
                    else VOCAB
                )
                or type(meta.get("chip_token_offset")) is not int
                or meta["chip_token_offset"] < 0
            ):
                raise ValueError("expected a merged resident grid or ADD32 checkpoint")
            resident_tilegrid = meta["layout"] in (
                RESIDENT_TILEGRID_LAYOUT,
                RESIDENT_GLOBALGRID_LAYOUT,
                RESIDENT_COMPUTE_LAYOUT,
            )
            resident_compute = meta["layout"] == RESIDENT_COMPUTE_LAYOUT
            resident_tilepipe = meta["layout"] in RESIDENT_TILEPIPE_LAYOUTS
            resident_add32 = meta["layout"] in RESIDENT_ADD32_LAYOUTS
            resident_loop32 = meta["layout"] in RESIDENT_LOOP32_LAYOUTS
            resident_branch32 = meta["layout"] in RESIDENT_BRANCH32_LAYOUTS
            resident_addmem32 = meta["layout"] in RESIDENT_ADDMEM32_LAYOUTS
            resident_shared32 = meta["layout"] in RESIDENT_SHARED32_LAYOUTS
            if args.resident_shared32 and not resident_shared32:
                raise ValueError("--resident-shared32 requires a shared32 checkpoint")
            if args.resident_add32 and not resident_add32:
                raise ValueError(
                    "--resident-add32 requires a resident ADD32 checkpoint"
                )
            if args.resident_addmem32 and not resident_addmem32:
                raise ValueError("--resident-addmem32 requires an addmem32 checkpoint")
            if args.resident_branch32 and not resident_branch32:
                raise ValueError("--resident-branch32 requires a branch32 checkpoint")
            if args.resident_loop32 and not resident_loop32:
                raise ValueError("--resident-loop32 requires a loop32 checkpoint")
            if args.resident_compute and not resident_compute:
                raise ValueError("--resident-compute requires a compute checkpoint")
            offset = meta["chip_token_offset"]
        if args.schedule_k and not resident_tilepipe:
            raise ValueError("--schedule-k requires a tilepipe checkpoint")
        if args.grid_2x2 and not resident_tilegrid:
            raise ValueError("--grid-2x2 requires a tilegrid checkpoint")
        if args.retain_c and not resident_add32:
            raise ValueError("--retain-c requires a resident ADD32 checkpoint")
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    device = args.device
    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    model, circuit = load_tile_model(
        args.model,
        device,
        resident_grid=True,
        resident_add32=resident_add32,
        resident_shared32=resident_shared32,
        resident_addmem32=resident_addmem32,
        resident_branch32=resident_branch32,
        resident_loop32=resident_loop32,
        resident_tilepipe=resident_tilepipe,
        resident_tilegrid=resident_tilegrid,
        resident_compute=resident_compute,
    )
    standalone = (
        "standalone-resident-compute"
        if resident_compute
        else "standalone-resident-tilegrid"
        if resident_tilegrid
        else "standalone-resident-tilepipe"
        if resident_tilepipe
        else "standalone-resident-loop32"
        if resident_loop32
        else "standalone-resident-branch32"
        if resident_branch32
        else "standalone-resident-addmem32"
        if resident_addmem32
        else "standalone-resident-shared32"
        if resident_shared32
        else "standalone-resident-add32"
    )
    if args.grid_2x2:
        from qwengpu.eval_tilegrid import execute_matrix

        execution = execute_matrix(model, circuit, a, b, c, device, offset=offset)
    elif args.schedule_k:
        execution = run_scheduled_matmul(model, circuit, device, offset, a, b, c)
    else:
        execution = run_matmul(
            model,
            circuit,
            device,
            offset,
            a,
            b,
            c,
            resident_add32=resident_add32,
            retain_c=args.retain_c,
        )
    result = {
        "format": FORMAT,
        "input_sha256": hashlib.sha256(payload).hexdigest(),
        "device": device,
        "torch": torch.__version__,
        "merge": meta,
        "transport_mode": "state_store_plan"
        if args.retain_c or args.schedule_k or args.grid_2x2
        else "fixed",
        **({"schedule_k": True} if args.schedule_k else {}),
        **({"retain_c": True} if args.retain_c else {}),
        **({"grid_2x2": True} if args.grid_2x2 else {}),
        **(
            {"checkpoint": meta["layout"] if meta else standalone}
            if resident_add32
            else {}
        ),
        **execution,
    }
    output = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
