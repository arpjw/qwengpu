"""Launch a two-warp resident QwenGPU kernel from JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from qwengpu.compiled_tile8 import VOCAB
from qwengpu.eval_tile8 import (
    LOOP_LAYOUT,
    RESIDENT_ADD32_LAYOUT,
    RESIDENT_DELTA_LAYOUT,
    RESIDENT_GRID_LAYOUT,
    RESIDENT_IO_LAYOUT,
    RESIDENT_PLAN_LAYOUT,
    RESIDENT_ROM_LAYOUT,
    RESIDENT_STATE_LAYOUT,
    RESIDENT_TRANSPORTS,
    RESIDENT_WIDE_LAYOUT,
    execute,
    load_tile_model,
    parse_launch,
)

RESULT_FORMAT = "qwengpu-resident-launch-v1"
EXIT_CODES = {"halted": 0, "deadlock": 3, "step_limit": 4}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", type=Path)
    parser.add_argument(
        "--model", type=Path, help="merged looping Qwen3.5-9B checkpoint"
    )
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument(
        "--resident-add32",
        action="store_true",
        help="enable the resident ADD32I opcode",
    )
    variants.add_argument(
        "--resident-wide",
        action="store_true",
        help="seed and report both 16-bit halves of each warp register",
    )
    variants.add_argument(
        "--resident-grid",
        action="store_true",
        help="let the model stop after the requested number of row pairs",
    )
    variants.add_argument(
        "--resident-io",
        action="store_true",
        help="read state and stores using model-generated plans",
    )
    variants.add_argument(
        "--resident-plan",
        action="store_true",
        help="read state using the model-generated plan",
    )
    variants.add_argument(
        "--resident-delta",
        action="store_true",
        help="relay only changed state in the cache",
    )
    variants.add_argument(
        "--resident-state",
        action="store_true",
        help="keep warp programs and register banks in the cache",
    )
    variants.add_argument(
        "--resident-rom",
        action="store_true",
        help="keep both programs in the launch cache",
    )
    parser.add_argument(
        "--transport",
        choices=tuple(mode.replace("_", "-") for mode in RESIDENT_TRANSPORTS),
        help="select the read protocol on a resident I/O checkpoint",
    )
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument(
        "--out", type=Path, help="write JSON to a file instead of stdout"
    )
    parser.add_argument(
        "--trace", action="store_true", help="include instruction and scheduler traces"
    )
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")

    offset = 0
    resident_add32 = args.resident_add32
    resident_wide = args.resident_wide or resident_add32
    resident_grid = args.resident_grid or resident_wide
    resident_io = args.resident_io or resident_grid
    resident_plan = args.resident_plan or resident_io
    resident_delta = args.resident_delta or resident_plan
    resident_state = args.resident_state or resident_delta
    resident_rom = args.resident_rom or resident_state
    layout = (
        "standalone-resident-add32"
        if resident_add32
        else "standalone-resident-wide"
        if resident_wide
        else "standalone-resident-grid"
        if resident_grid
        else "standalone-resident-io"
        if resident_io
        else "standalone-resident-plan"
        if resident_plan
        else "standalone-resident-delta"
        if resident_delta
        else "standalone-resident-state"
        if resident_state
        else "standalone-resident-rom"
        if resident_rom
        else "standalone-looping"
    )
    try:
        payload = args.kernel.read_bytes()
        if args.model is not None:
            meta = json.loads(
                (args.model / "qwengpu9b.json").read_text(encoding="utf-8")
            )
            if (
                not isinstance(meta, dict)
                or meta.get("layout")
                not in (
                    LOOP_LAYOUT,
                    RESIDENT_ROM_LAYOUT,
                    RESIDENT_STATE_LAYOUT,
                    RESIDENT_DELTA_LAYOUT,
                    RESIDENT_PLAN_LAYOUT,
                    RESIDENT_IO_LAYOUT,
                    RESIDENT_GRID_LAYOUT,
                    RESIDENT_WIDE_LAYOUT,
                    RESIDENT_ADD32_LAYOUT,
                )
                or meta.get("chip_vocab") != VOCAB
                or type(meta.get("chip_token_offset")) is not int
                or meta["chip_token_offset"] < 0
            ):
                raise ValueError("expected a merged looping Qwen3.5-9B checkpoint")
            layout = meta["layout"]
            resident_add32 = layout == RESIDENT_ADD32_LAYOUT
            resident_wide = resident_add32 or layout == RESIDENT_WIDE_LAYOUT
            resident_grid = resident_wide or layout == RESIDENT_GRID_LAYOUT
            resident_io = resident_grid or layout == RESIDENT_IO_LAYOUT
            resident_plan = resident_io or layout == RESIDENT_PLAN_LAYOUT
            resident_delta = resident_plan or layout == RESIDENT_DELTA_LAYOUT
            resident_state = resident_delta or layout == RESIDENT_STATE_LAYOUT
            resident_rom = resident_state or layout == RESIDENT_ROM_LAYOUT
            if args.resident_rom and not resident_rom:
                raise ValueError("--resident-rom requires a resident ROM checkpoint")
            if args.resident_state and not resident_state:
                raise ValueError(
                    "--resident-state requires a resident state checkpoint"
                )
            if args.resident_delta and not resident_delta:
                raise ValueError(
                    "--resident-delta requires a resident delta checkpoint"
                )
            if args.resident_plan and not resident_plan:
                raise ValueError("--resident-plan requires a resident plan checkpoint")
            if args.resident_io and not resident_io:
                raise ValueError("--resident-io requires a resident I/O checkpoint")
            if args.resident_grid and not resident_grid:
                raise ValueError("--resident-grid requires a resident grid checkpoint")
            if args.resident_wide and not resident_wide:
                raise ValueError("--resident-wide requires a resident wide checkpoint")
            if args.resident_add32 and not resident_add32:
                raise ValueError(
                    "--resident-add32 requires a resident ADD32 checkpoint"
                )
            offset = meta["chip_token_offset"]
        if args.transport and not resident_io:
            raise ValueError("--transport requires a resident I/O checkpoint")
        launch = parse_launch(
            json.loads(payload),
            looping=True,
            resident_grid=resident_grid,
            resident_wide=resident_wide,
            resident_add32=resident_add32,
        )
    except (OSError, ValueError) as exc:
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
        looping=True,
        resident_rom=resident_rom,
        resident_state=resident_state,
        resident_delta=resident_delta,
        resident_plan=resident_plan,
        resident_io=resident_io,
        resident_grid=resident_grid,
        resident_wide=resident_wide,
        resident_add32=resident_add32,
    )
    execution = (
        RESIDENT_TRANSPORTS[args.transport.replace("-", "_")]
        if args.transport
        else {
            "looping": True,
            "resident_rom": resident_rom,
            "resident_state": resident_state,
            "resident_delta": resident_delta,
            "resident_plan": resident_plan,
            "resident_io": resident_io,
        }
    )
    observed = execute(
        model,
        circuit,
        launch,
        device,
        args.max_steps,
        offset,
        resident_grid=resident_grid,
        resident_wide=resident_wide,
        resident_add32=resident_add32,
        **execution,
    )
    trace = observed.pop("trace")
    schedules = observed.pop("schedules")
    result = {
        "format": RESULT_FORMAT,
        "checkpoint": layout,
        "device": device,
        "kernel_sha256": hashlib.sha256(payload).hexdigest(),
        "max_steps": args.max_steps,
        "instructions": len(trace),
        **observed,
    }
    if resident_grid:
        result["first_row_pair"] = launch["first_row_pair"]
        result["row_pairs"] = launch["row_pairs"]
    if args.transport:
        result["transport_mode"] = args.transport
    if args.trace:
        result.update(trace=trace, schedules=schedules)
    output = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)
    transport = observed["transport"]
    if (
        transport["fresh_caches"] != 1
        or not transport["resident_memory_exact"]
        or (resident_grid and not transport["resident_grid_exact"])
        or (resident_rom and not transport["resident_rom_exact"])
        or (resident_state and not transport["resident_state_exact"])
    ):
        return 1
    return EXIT_CODES[observed["outcome"]]


if __name__ == "__main__":
    raise SystemExit(main())
