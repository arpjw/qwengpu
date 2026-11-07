"""Compare fixed and planned transport for common kernels on one CUDA model."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from qwengpu.compiled_tile8 import RESIDENT_COMPUTE_LAYOUT
from qwengpu.eval_compute import FIXTURE
from qwengpu.eval_tile8 import load_tile_model
from qwengpu.run_compute import execute_request, parse_request

MODES = ("fixed", "state_store_plan")


def digest(result: dict) -> str:
    # Transport and its per-forward audit differ; machine execution must not.
    execution = {
        "output": result["output"],
        "global_words": result["global_words"],
        "launches": [
            {
                key: value
                for key, value in run.items()
                if key not in ("transport", "audit")
            }
            for run in result["launches"]
        ],
    }
    payload = json.dumps(execution, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    if args.rounds < 2:
        parser.error("use at least two alternating rounds")
    meta = json.loads((args.model / "qwengpu9b.json").read_text())
    if meta["layout"] != RESIDENT_COMPUTE_LAYOUT:
        parser.error("expected a merged compute checkpoint")
    payload = FIXTURE.read_bytes()
    inputs = json.loads(payload)
    names = ("vector_17", "matmul_3x3x3")
    model, circuit = load_tile_model(args.model, "cuda", resident_compute=True)
    records, reference = [], {}
    for round_index in range(-1, args.rounds):
        for name in names:
            request = parse_request(inputs[name])
            for mode in MODES[:: -1 if round_index % 2 else 1]:
                calls = 0

                def count_forward(_model, _args) -> None:
                    nonlocal calls
                    calls += 1

                hook = model.register_forward_pre_hook(count_forward)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                try:
                    result = execute_request(
                        model,
                        circuit,
                        request,
                        "cuda",
                        meta["chip_token_offset"],
                        transport_mode=mode,
                    )
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                finally:
                    hook.remove()
                fingerprint = digest(result)
                if fingerprint != reference.setdefault(name, fingerprint):
                    raise ValueError("transport modes changed kernel execution")
                if calls != sum(
                    run["transport"]["forward_calls"] for run in result["launches"]
                ):
                    raise ValueError("model hook and transport counters disagree")
                record = {
                    "name": name,
                    "mode": mode,
                    "round": round_index,
                    "warmup": round_index == -1,
                    "seconds": elapsed,
                    "forward_calls": calls,
                    "execution_sha256": fingerprint,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "transport": [run["transport"] for run in result["launches"]],
                    "exact": result["exact"],
                }
                records.append(record)
                print(
                    json.dumps(
                        {
                            key: record[key]
                            for key in (
                                "name",
                                "mode",
                                "round",
                                "seconds",
                                "forward_calls",
                                "exact",
                            )
                        }
                    ),
                    flush=True,
                )
    summaries = {}
    for name in names:
        summaries[name] = {
            mode: {
                "median_seconds": statistics.median(
                    row["seconds"]
                    for row in records
                    if row["name"] == name and row["mode"] == mode and not row["warmup"]
                ),
            }
            for mode in MODES
        }
    report = {
        "merge": meta,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "fixture_sha256": hashlib.sha256(payload).hexdigest(),
        "rounds": args.rounds,
        "records": records,
        "summary": summaries,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
