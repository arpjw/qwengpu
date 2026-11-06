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
    RESIDENT_DELTA_LAYOUT,
    RESIDENT_ROM_LAYOUT,
    RESIDENT_STATE_LAYOUT,
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
    resident_delta = args.resident_delta
    resident_state = args.resident_state or resident_delta
    resident_rom = args.resident_rom or resident_state
    layout = (
        "standalone-resident-delta"
        if resident_delta
        else "standalone-resident-state"
        if resident_state
        else "standalone-resident-rom"
        if resident_rom
        else "standalone-looping"
    )
    try:
        payload = args.kernel.read_bytes()
        launch = parse_launch(json.loads(payload), looping=True)
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
                )
                or meta.get("chip_vocab") != VOCAB
                or type(meta.get("chip_token_offset")) is not int
                or meta["chip_token_offset"] < 0
            ):
                raise ValueError("expected a merged looping Qwen3.5-9B checkpoint")
            layout = meta["layout"]
            resident_delta = layout == RESIDENT_DELTA_LAYOUT
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
            offset = meta["chip_token_offset"]
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
    )
    observed = execute(
        model,
        circuit,
        launch,
        device,
        args.max_steps,
        offset,
        looping=True,
        resident_rom=resident_rom,
        resident_state=resident_state,
        resident_delta=resident_delta,
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
        or (resident_rom and not transport["resident_rom_exact"])
        or (resident_state and not transport["resident_state_exact"])
    ):
        return 1
    return EXIT_CODES[observed["outcome"]]


if __name__ == "__main__":
    raise SystemExit(main())
