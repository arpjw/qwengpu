"""Compare resident transport modes on a single merged CUDA checkpoint."""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import importlib.metadata
import json
import os
import pstats
import statistics
import time
from pathlib import Path

import torch
import transformers

from qwengpu.eval_tile8 import (
    RESIDENT_IO_LAYOUT,
    _parse_case,
    execute,
    load_tile_model,
    oracle,
)

MODES = {
    "fixed": {"resident_delta": True},
    "state_plan": {"resident_plan": True},
    "state_store_plan": {"resident_io": True},
}
ORDERS = (
    ("fixed", "state_plan", "state_store_plan"),
    ("state_plan", "state_store_plan", "fixed"),
    ("state_store_plan", "fixed", "state_plan"),
    ("fixed", "state_store_plan", "state_plan"),
    ("state_store_plan", "state_plan", "fixed"),
    ("state_plan", "fixed", "state_store_plan"),
)


def _digest(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _write(path: Path, report: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _trial(model, circuit, case, offset, max_steps, mode, profile) -> tuple[dict, dict]:
    calls = 0
    events = []

    def before_forward(_model, _args) -> None:
        nonlocal calls
        calls += 1
        if profile:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            events.append((start, end))

    def after_forward(_model, _args, _result) -> None:
        events[-1][1].record()

    pre = model.register_forward_pre_hook(before_forward)
    post = model.register_forward_hook(after_forward) if profile else None
    cpu_profile = cProfile.Profile() if profile else None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base_bytes = torch.cuda.memory_allocated()
    start = time.perf_counter()
    try:
        if cpu_profile is not None:
            cpu_profile.enable()
        observed = execute(
            model, circuit, case, "cuda", max_steps, offset, **MODES[mode]
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    finally:
        if cpu_profile is not None:
            cpu_profile.disable()
        pre.remove()
        if post is not None:
            post.remove()

    transport = observed.pop("transport")
    if "forward_calls" in transport and transport["forward_calls"] != calls:
        raise ValueError("forward hook and transport counter disagree")
    record = {
        "mode": mode,
        "seconds": elapsed,
        "forward_calls": calls,
        "allocated_before_bytes": base_bytes,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "transport": transport,
        "instructions": len(observed["trace"]),
        "outcome": observed["outcome"],
        "trace_sha256": _digest(observed),
    }
    if cpu_profile is not None:
        stats = pstats.Stats(cpu_profile)
        largest = sorted(stats.stats.items(), key=lambda row: row[1][3], reverse=True)[
            :12
        ]
        record["profile"] = {
            "forward_stream_seconds": sum(
                start.elapsed_time(end) for start, end in events
            )
            / 1000,
            "top_cpu_cumulative": [
                {
                    "function": f"{Path(filename).name}:{line}:{name}",
                    "seconds": totals[3],
                }
                for (filename, line, name), totals in largest
            ],
        }
    return record, observed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=6)
    args = parser.parse_args()
    if args.rounds < len(ORDERS) or args.rounds % len(ORDERS):
        parser.error("--rounds must be a positive multiple of six")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")

    payload = args.kernel.read_bytes()
    case = _parse_case(json.loads(payload), looping=True)
    meta = json.loads((args.model / "qwengpu9b.json").read_text(encoding="utf-8"))
    if meta["layout"] != RESIDENT_IO_LAYOUT:
        parser.error("expected the merged resident I/O checkpoint")
    max_steps = 256
    expected = oracle(case, max_steps, looping=True)
    if expected["outcome"] != case["expected_outcome"] or (
        case["expected_matrix"] is not None
        and expected["matrix"] != case["expected_matrix"]
    ):
        raise ValueError("fixture outcome or matrix disagrees with the reference")
    expected_digest = _digest(expected)
    device = torch.cuda.get_device_properties(0)
    report = {
        "fixture": args.kernel.name,
        "fixture_sha256": hashlib.sha256(payload).hexdigest(),
        "checkpoint": meta,
        "max_steps": max_steps,
        "gpu": device.name,
        "gpu_bytes": device.total_memory,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "flash_linear_attention": importlib.metadata.version("flash-linear-attention"),
        "torch_threads": torch.get_num_threads(),
        "available_cpu_cores": len(os.sched_getaffinity(0)),
        "initial_load_average": os.getloadavg(),
        "expected_trace_sha256": expected_digest,
        "rounds": args.rounds,
        "warmups": [],
        "trials": [],
        "failures": [],
        "complete": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write(args.out, report)
    started = time.perf_counter()
    model, circuit = load_tile_model(args.model, "cuda", resident_io=True)
    torch.cuda.synchronize()
    report["checkpoint_load_seconds"] = time.perf_counter() - started
    offset = meta["chip_token_offset"]

    def run(mode: str, round_number: int) -> None:
        try:
            record, observed = _trial(
                model, circuit, case, offset, max_steps, mode, round_number == 0
            )
            record["round"] = round_number
            transport = record["transport"]
            record["exact"] = (
                observed == expected
                and record["trace_sha256"] == expected_digest
                and transport["fresh_caches"] == 1
                and all(
                    transport[key]
                    for key in (
                        "resident_memory_exact",
                        "resident_rom_exact",
                        "resident_state_exact",
                    )
                )
            )
            report["warmups" if round_number == 0 else "trials"].append(record)
            if not record["exact"]:
                raise ValueError("launch differs from the reference trace")
        except Exception as exc:
            report["failures"].append(
                {"mode": mode, "round": round_number, "error": str(exc)}
            )
            _write(args.out, report)
            raise
        _write(args.out, report)
        print(
            json.dumps(
                {key: record[key] for key in ("mode", "round", "seconds", "exact")}
            ),
            flush=True,
        )

    for mode in MODES:
        run(mode, 0)
    for round_number in range(1, args.rounds + 1):
        for mode in ORDERS[(round_number - 1) % len(ORDERS)]:
            run(mode, round_number)

    report["summary"] = {}
    for mode in MODES:
        trials = [trial for trial in report["trials"] if trial["mode"] == mode]
        seconds = [trial["seconds"] for trial in trials]
        peaks = [trial["peak_allocated_bytes"] for trial in trials]
        report["summary"][mode] = {
            "trials": len(trials),
            "median_seconds": statistics.median(seconds),
            "range_seconds": [min(seconds), max(seconds)],
            "median_peak_allocated_bytes": int(statistics.median(peaks)),
            "range_peak_allocated_bytes": [min(peaks), max(peaks)],
        }
    report["complete"] = True
    _write(args.out, report)


if __name__ == "__main__":
    main()
