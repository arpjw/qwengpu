"""Launch a two-warp resident QwenGPU kernel from JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from qwengpu.alu32 import ALU_VOCAB
from qwengpu.compiled_tile8 import (
    ROM32_VOCAB,
    RESIDENT_COMPUTE_LAYOUT,
    RESIDENT_GLOBALGRID_LAYOUT,
    RESIDENT_TILEGRID_LAYOUT,
    TILEGRID_VOCAB,
    VOCAB,
)
from qwengpu.eval_tile8 import (
    LOOP_LAYOUT,
    RESIDENT_ADD32_LAYOUTS,
    RESIDENT_ADDMEM32_LAYOUTS,
    RESIDENT_BRANCH32_LAYOUTS,
    RESIDENT_DELTA_LAYOUT,
    RESIDENT_GRID_LAYOUT,
    RESIDENT_IO_LAYOUT,
    RESIDENT_LOOP32_LAYOUTS,
    RESIDENT_PLAN_LAYOUT,
    RESIDENT_ROM_LAYOUT,
    RESIDENT_SHARED32_LAYOUTS,
    RESIDENT_STATE_LAYOUT,
    RESIDENT_TILEPIPE_LAYOUTS,
    RESIDENT_TRANSPORTS,
    RESIDENT_WIDE_LAYOUT,
    execute,
    load_tile_model,
    oracle,
    parse_launch,
)
from qwengpu.eval_resident_generation import evaluate, load_generation_model
from qwengpu.memory_geometry import (
    FULL_INTEGER_LAYOUTS,
    INTERLEAVED_LAYOUTS,
    SCOPED_LAYOUTS,
    ALU32_LAYOUTS,
    MEMORY512_LAYOUTS,
    ROM32_LAYOUTS,
    WIDE_MEMORY,
)
from qwengpu.resident_generation import LAYOUT as GENERATION_LAYOUT

RESULT_FORMAT = "qwengpu-resident-launch-v1"
EXIT_CODES = {"halted": 0, "deadlock": 3, "step_limit": 4}


def write_result(result, destination):
    output = json.dumps(result, indent=2)
    if destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_generated(args, payload, meta):
    if meta is not None and (
        meta.get("layout") != GENERATION_LAYOUT
        or meta.get("chip_vocab") != ALU_VOCAB
        or type(meta.get("chip_token_offset")) is not int
        or meta["chip_token_offset"] < 0
    ):
        raise ValueError("expected a merged resident generation checkpoint")
    if args.transport:
        raise ValueError("--transport is for legacy resident I/O checkpoints")
    raw = json.loads(payload)
    case = parse_launch(raw, full_integer=True)
    device = args.device or default_device()
    model, circuit, offset = load_generation_model(args.model, device)
    _, observed = evaluate(model, circuit, case, device, offset, max_steps=args.max_steps)
    if "expected_outcome" in raw:
        observed["fixture_outcome_exact"] = observed["outcome"] == raw["expected_outcome"]
    exact = {key: value for key, value in observed.items() if key.endswith("_exact")}
    tokens = observed.pop("generated_tokens")
    expected = observed.pop("expected_tokens")
    result = {
        "format": RESULT_FORMAT,
        "checkpoint": GENERATION_LAYOUT if meta else "standalone-resident-generate",
        "device": device,
        "kernel_sha256": hashlib.sha256(payload).hexdigest(),
        "max_steps": args.max_steps,
        "instructions": observed.pop("steps"),
        "transport_mode": "stock-generation",
        "generated_token_count": len(tokens),
        **observed,
    }
    if args.trace:
        result.update(generated_tokens=tokens, expected_tokens=expected)
    if args.check_reference:
        result["reference"] = exact
    write_result(result, args.out)
    return EXIT_CODES.get(observed["outcome"], 1) if all(exact.values()) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", type=Path)
    parser.add_argument(
        "--model", type=Path, help="merged looping Qwen3.5-9B checkpoint"
    )
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument("--resident-memory512", action="store_true")
    variants.add_argument("--resident-rom32", action="store_true")
    variants.add_argument("--resident-alu32", action="store_true")
    variants.add_argument("--resident-scoped", action="store_true")
    variants.add_argument("--resident-interleaved", action="store_true")
    variants.add_argument(
        "--resident-compute", action="store_true", help="use the common compute ISA"
    )
    variants.add_argument(
        "--resident-globalgrid",
        action="store_true",
        help="use programmable workgroups and global memory",
    )
    variants.add_argument(
        "--resident-tilegrid",
        action="store_true",
        help="use the standalone tilegrid circuit in single-workgroup mode",
    )
    variants.add_argument(
        "--resident-tilepipe",
        action="store_true",
        help="use the standalone circuit from the final tilepipe checkpoint",
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
        help="enable indexed high-limb loads and stores",
    )
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
    parser.add_argument(
        "--check-reference",
        action="store_true",
        help="compare the execution trace and final state to the reference interpreter",
    )
    variants.add_argument("--full-integer", action="store_true")
    variants.add_argument("--resident-generate", action="store_true")
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")

    offset = 0
    full_integer = args.full_integer
    resident_interleaved = args.resident_interleaved or full_integer
    resident_scoped = args.resident_scoped or resident_interleaved
    resident_alu32 = args.resident_alu32 or resident_scoped
    resident_rom32 = args.resident_rom32 or resident_alu32
    resident_memory512 = args.resident_memory512 or resident_rom32
    resident_compute = args.resident_compute or resident_memory512
    resident_globalgrid = args.resident_globalgrid or resident_compute
    resident_tilegrid = args.resident_tilegrid or resident_globalgrid
    resident_tilepipe = args.resident_tilepipe or resident_tilegrid
    resident_loop32 = args.resident_loop32 or resident_tilepipe
    resident_branch32 = args.resident_branch32 or resident_loop32
    resident_addmem32 = args.resident_addmem32 or resident_branch32
    resident_shared32 = args.resident_shared32 or resident_addmem32
    resident_add32 = args.resident_add32 or resident_shared32
    resident_wide = args.resident_wide or resident_add32
    resident_grid = args.resident_grid or resident_wide
    resident_io = args.resident_io or resident_grid
    resident_plan = args.resident_plan or resident_io
    resident_delta = args.resident_delta or resident_plan
    resident_state = args.resident_state or resident_delta
    resident_rom = args.resident_rom or resident_state
    layout = (
        "standalone-full-integer"
        if full_integer
        else "standalone-resident-interleaved"
        if resident_interleaved
        else "standalone-resident-scoped"
        if resident_scoped
        else "standalone-resident-alu32"
        if resident_alu32
        else "standalone-resident-rom32"
        if resident_rom32
        else "standalone-resident-memory512"
        if resident_memory512
        else "standalone-resident-compute"
        if resident_compute
        else "standalone-resident-globalgrid"
        if resident_globalgrid
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
        meta = json.loads(
            (args.model / "qwengpu9b.json").read_text(encoding="utf-8")
        ) if args.model is not None else None
        if args.model is not None and not isinstance(meta, dict):
            raise ValueError("expected checkpoint metadata")
        if args.resident_generate or (meta is not None and meta.get("layout") == GENERATION_LAYOUT):
            return run_generated(args, payload, meta)
        if meta is not None:
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
                    *RESIDENT_ADD32_LAYOUTS,
                )
                or meta.get("chip_vocab")
                != (
                    ALU_VOCAB
                    if meta.get("layout") in ALU32_LAYOUTS
                    else ROM32_VOCAB
                    if meta.get("layout") in ROM32_LAYOUTS
                    else WIDE_MEMORY.vocab
                    if meta.get("layout") in MEMORY512_LAYOUTS
                    else TILEGRID_VOCAB
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
                raise ValueError("expected a merged looping Qwen3.5-9B checkpoint")
            layout = meta["layout"]
            resident_tilepipe = layout in RESIDENT_TILEPIPE_LAYOUTS
            full_integer = layout in FULL_INTEGER_LAYOUTS
            resident_interleaved = layout in INTERLEAVED_LAYOUTS
            resident_scoped = layout in SCOPED_LAYOUTS
            resident_alu32 = layout in ALU32_LAYOUTS
            resident_rom32 = layout in ROM32_LAYOUTS
            resident_memory512 = layout in MEMORY512_LAYOUTS
            resident_compute = layout in (RESIDENT_COMPUTE_LAYOUT, *MEMORY512_LAYOUTS)
            resident_globalgrid = layout in (
                RESIDENT_GLOBALGRID_LAYOUT,
                RESIDENT_COMPUTE_LAYOUT,
                *MEMORY512_LAYOUTS,
            )
            resident_tilegrid = layout in (
                RESIDENT_TILEGRID_LAYOUT,
                RESIDENT_GLOBALGRID_LAYOUT,
                RESIDENT_COMPUTE_LAYOUT,
                *MEMORY512_LAYOUTS,
            )
            resident_loop32 = layout in RESIDENT_LOOP32_LAYOUTS
            resident_branch32 = layout in RESIDENT_BRANCH32_LAYOUTS
            resident_addmem32 = layout in RESIDENT_ADDMEM32_LAYOUTS
            resident_shared32 = layout in RESIDENT_SHARED32_LAYOUTS
            resident_add32 = layout in RESIDENT_ADD32_LAYOUTS
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
            if args.resident_shared32 and not resident_shared32:
                raise ValueError("--resident-shared32 requires a shared32 checkpoint")
            if args.resident_addmem32 and not resident_addmem32:
                raise ValueError("--resident-addmem32 requires an addmem32 checkpoint")
            if args.resident_branch32 and not resident_branch32:
                raise ValueError("--resident-branch32 requires a branch32 checkpoint")
            if args.resident_loop32 and not resident_loop32:
                raise ValueError("--resident-loop32 requires a loop32 checkpoint")
            if args.resident_tilepipe and not resident_tilepipe:
                raise ValueError("--resident-tilepipe requires a tilepipe checkpoint")
            if args.resident_tilegrid and not resident_tilegrid:
                raise ValueError("--resident-tilegrid requires a tilegrid checkpoint")
            if args.resident_globalgrid and not resident_globalgrid:
                raise ValueError(
                    "--resident-globalgrid requires a globalgrid checkpoint"
                )
            if args.full_integer and not full_integer:
                raise ValueError("--full-integer requires a full integer checkpoint")
            if args.resident_interleaved and not resident_interleaved:
                raise ValueError(
                    "--resident-interleaved requires an interleaved checkpoint"
                )
            if args.resident_scoped and not resident_scoped:
                raise ValueError("--resident-scoped requires a scoped checkpoint")
            if args.resident_alu32 and not resident_alu32:
                raise ValueError("--resident-alu32 requires an ALU32 checkpoint")
            if args.resident_rom32 and not resident_rom32:
                raise ValueError("--resident-rom32 requires a ROM32 checkpoint")
            if args.resident_memory512 and not resident_memory512:
                raise ValueError("--resident-memory512 requires a memory512 checkpoint")
            if args.resident_compute and not resident_compute:
                raise ValueError("--resident-compute requires a compute checkpoint")
            offset = meta["chip_token_offset"]
        if args.transport and not resident_io:
            raise ValueError("--transport requires a resident I/O checkpoint")
        launch = parse_launch(
            json.loads(payload),
            looping=True,
            resident_grid=resident_grid,
            resident_wide=resident_wide,
            resident_add32=resident_add32,
            resident_shared32=resident_shared32,
            resident_addmem32=resident_addmem32,
            resident_branch32=resident_branch32,
            resident_loop32=resident_loop32,
            resident_globalgrid=resident_globalgrid,
            resident_compute=resident_compute,
            resident_memory512=resident_memory512,
            resident_rom32=resident_rom32,
            resident_alu32=resident_alu32,
            resident_scoped=resident_scoped,
            resident_interleaved=resident_interleaved,
            full_integer=full_integer,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    device = args.device or default_device()
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
        resident_shared32=resident_shared32,
        resident_addmem32=resident_addmem32,
        resident_branch32=resident_branch32,
        resident_loop32=resident_loop32,
        resident_tilepipe=resident_tilepipe,
        resident_tilegrid=resident_tilegrid,
        resident_globalgrid=resident_globalgrid,
        resident_compute=resident_compute,
        resident_memory512=resident_memory512,
        resident_rom32=resident_rom32,
        resident_alu32=resident_alu32,
        resident_scoped=resident_scoped,
        resident_interleaved=resident_interleaved,
        full_integer=full_integer,
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
    if args.check_reference:
        expected = oracle(launch, args.max_steps, looping=True)
        reference = {key: observed.get(key) == value for key, value in expected.items()}
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
    if args.check_reference:
        result["reference"] = reference
    write_result(result, args.out)
    transport = observed["transport"]
    if (
        transport["fresh_caches"] != 1
        or not transport["resident_memory_exact"]
        or (resident_grid and not transport["resident_grid_exact"])
        or (resident_rom and not transport["resident_rom_exact"])
        or (resident_state and not transport["resident_state_exact"])
        or (
            resident_scoped
            and "scoped_words" in launch
            and not transport["resident_scoped_exact"]
        )
        or (args.check_reference and not all(reference.values()))
    ):
        return 1
    return EXIT_CODES[observed["outcome"]]


if __name__ == "__main__":
    raise SystemExit(main())
