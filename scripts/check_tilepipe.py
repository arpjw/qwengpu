"""Verify one scheduled two-slice launch and its inherited GPU traces."""

import argparse
import json
from pathlib import Path

from check_resident_add32_full import RESULTS, cases_by_name, load_report

LAYOUT = "qwengpu-resident-tilepipe-9b-v1"
NEXT_BANK = 1632


def check_launch(report: dict) -> None:
    baseline = load_report("resident_tilebanks_cached_tile_mps")
    assert report["input_sha256"] == baseline["input_sha256"]
    assert report["outcome"] == "halted"
    assert len(report["trace"]) == 232 and len(report["schedules"]) == 233
    boundary = report["bank_transition"]
    assert boundary == {
        "ticket": NEXT_BANK,
        "instructions": 116,
        "schedule_index": 116,
        "matrix": baseline["passes"][0]["matrix"],
        "resident_memory_exact": True,
        "bank_before": 0,
    }
    tickets = [step["ticket"] for step in report["schedules"]]
    assert tickets.count(NEXT_BANK) == 1 and tickets[116] == NEXT_BANK
    assert report["schedules"][116]["controls"] == [1024, 1088]
    for actual, old in zip(report["passes"], baseline["passes"], strict=True):
        assert actual == {key: old[key] for key in actual}
    assert report["matrix"] == baseline["passes"][1]["matrix"]

    transport = report["transport"]
    assert transport["fresh_caches"] == 1
    assert transport["matrix_initial_tokens"] == 768
    assert transport["matrix_patch_tokens"] == 512
    assert transport["rom_program_tokens"] == 32
    assert all(
        transport[f"resident_{key}_exact"]
        for key in ("grid", "memory", "rom", "state", "operand_banks")
    )
    calls = report["memory_tokens_by_call"]
    assert calls[0] == {
        "operands": 256,
        "alternate_operands": 256,
        "c": 256,
        "grid": 1,
        "transition": 0,
    }
    assert all(
        not any(call[key] for key in ("operands", "alternate_operands", "grid"))
        for call in calls[1:]
    )
    assert sum(call["transition"] for call in calls) == 1
    assert sum(call["c"] for call in calls[1:]) == 512


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="*", type=Path)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    paths = args.reports or [
        RESULTS / f"resident_tilepipe_{backend}.json"
        for backend in ("mps", "9b_cuda_fla")
    ]
    previous = None
    for path in paths:
        report = json.loads(path.read_text())
        check_launch(report)
        if report["merge"] is not None:
            assert report["merge"]["layout"] == LAYOUT
        if previous is not None:
            for key in (
                "bank_transition",
                "passes",
                "trace",
                "schedules",
                "transport",
                "memory_tokens_by_call",
            ):
                assert report[key] == previous[key]
        previous = report
        print(
            f"{path.name}: exact two-slice launch with one model-generated transition"
        )
    if args.full:
        expected = cases_by_name(load_report("resident_tilebanks_full_mps"))
        for backend in ("mps", "9b_cuda_fla"):
            report = load_report(f"resident_tilepipe_full_{backend}")
            assert cases_by_name(report) == expected
            if report["merge"] is not None:
                assert report["merge"] == previous["merge"]
            print(f"{backend}: exact records for all 27 inherited cases")


if __name__ == "__main__":
    main()
