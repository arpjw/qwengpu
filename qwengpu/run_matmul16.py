"""Compute C + A @ B modulo 65536 with eight resident grid launches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from qwengpu.compiled_tile8 import VOCAB, WARP_DIRTY_OFFSET
from qwengpu.eval_tile import _unsigned
from qwengpu.eval_tile8 import (
    RESIDENT_ADD32_LAYOUTS,
    RESIDENT_ADDMEM32_LAYOUTS,
    RESIDENT_BRANCH32_LAYOUT,
    RESIDENT_GRID_LAYOUT,
    RESIDENT_SHARED32_LAYOUTS,
    RESIDENT_TRANSPORTS,
    execute,
    launch_cached_tile,
    load_tile_model,
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
        "--resident-add32",
        action="store_true",
        help="use the standalone checkpoint that also runs ADD32I",
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
    parser.add_argument(
        "--retain-c", action="store_true", help="keep each output tile in one cache"
    )
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument(
        "--out", type=Path, help="write JSON to a file instead of stdout"
    )
    args = parser.parse_args()

    meta = None
    offset = 0
    resident_branch32 = args.resident_branch32
    resident_addmem32 = args.resident_addmem32 or resident_branch32
    resident_shared32 = args.resident_shared32 or resident_addmem32
    resident_add32 = args.resident_add32 or resident_shared32
    try:
        payload = args.input.read_bytes()
        a, b, c = parse_inputs(json.loads(payload))
        if args.model:
            meta = json.loads((args.model / "qwengpu9b.json").read_text())
            if (
                not isinstance(meta, dict)
                or meta.get("layout")
                not in (RESIDENT_GRID_LAYOUT, *RESIDENT_ADD32_LAYOUTS)
                or meta.get("chip_vocab") != VOCAB
                or type(meta.get("chip_token_offset")) is not int
                or meta["chip_token_offset"] < 0
            ):
                raise ValueError("expected a merged resident grid or ADD32 checkpoint")
            resident_add32 = meta["layout"] in RESIDENT_ADD32_LAYOUTS
            resident_branch32 = meta["layout"] == RESIDENT_BRANCH32_LAYOUT
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
            offset = meta["chip_token_offset"]
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
    )
    standalone = (
        "standalone-resident-branch32"
        if resident_branch32
        else "standalone-resident-addmem32"
        if resident_addmem32
        else "standalone-resident-shared32"
        if resident_shared32
        else "standalone-resident-add32"
    )
    result = {
        "format": FORMAT,
        "input_sha256": hashlib.sha256(payload).hexdigest(),
        "device": device,
        "torch": torch.__version__,
        "merge": meta,
        "transport_mode": "state_store_plan" if args.retain_c else "fixed",
        **({"retain_c": True} if args.retain_c else {}),
        **(
            {"checkpoint": meta["layout"] if meta else standalone}
            if resident_add32
            else {}
        ),
        **run_matmul(
            model,
            circuit,
            device,
            offset,
            a,
            b,
            c,
            resident_add32=resident_add32,
            retain_c=args.retain_c,
        ),
    }
    output = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
