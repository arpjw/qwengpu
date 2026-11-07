"""Recompute benchmark summaries and match execution to frozen model evidence."""

from __future__ import annotations

import hashlib
import json
import math
import statistics

from check_compute import FIXTURE
from check_resident_add32_full import load_report


def main() -> None:
    report = load_report("compute_transport_9b_cuda_fla")
    reference = load_report("compute_9b_cuda_fla")
    assert report["merge"] == reference["merge"]
    assert report["fixture_sha256"] == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    names = {"vector_17", "matmul_3x3x3"}
    modes = {"fixed", "state_store_plan"}
    rounds = report["rounds"]
    assert rounds >= 2 and len(report["records"]) == len(names) * len(modes) * (
        rounds + 1
    )
    assert report["summary"].keys() == names
    cases = {case["name"]: case for case in reference["cases"]}
    for name in names:
        case = cases[name]
        execution = {
            "output": case["output"],
            "global_words": case["global_words"],
            "launches": [
                {
                    key: value
                    for key, value in run.items()
                    if key not in ("transport", "audit")
                }
                for run in case["launches"]
            ],
        }
        payload = json.dumps(execution, sort_keys=True, separators=(",", ":")).encode()
        fingerprint = hashlib.sha256(payload).hexdigest()
        assert report["summary"][name].keys() == modes
        for mode in modes:
            rows = [
                r for r in report["records"] if (r["name"], r["mode"]) == (name, mode)
            ]
            assert len(rows) == rounds + 1
            assert {row["round"] for row in rows} == set(range(-1, rounds))
            for row in rows:
                assert row["warmup"] == (row["round"] == -1)
                assert row["exact"] and row["execution_sha256"] == fingerprint
                assert math.isfinite(row["seconds"]) and row["seconds"] > 0
                assert row["forward_calls"] > 0 and row["peak_allocated_bytes"] > 0
                if mode == "state_store_plan":
                    assert row["transport"] == [
                        run["transport"] for run in case["launches"]
                    ]
            trials = [row for row in rows if not row["warmup"]]
            assert len({row["forward_calls"] for row in trials}) == 1
            median = statistics.median(row["seconds"] for row in trials)
            assert report["summary"][name][mode]["median_seconds"] == median
        fixed = report["summary"][name]["fixed"]["median_seconds"]
        planned = report["summary"][name]["state_store_plan"]["median_seconds"]
        print(
            f"{name}: exact execution; fixed {fixed:.3f}s, planned {planned:.3f}s, {100 * (1 - fixed / planned):.1f}% lower observed latency"
        )


if __name__ == "__main__":
    main()
