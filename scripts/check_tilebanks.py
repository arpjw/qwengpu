"""Verify preloaded operand banks and replay the loop32 qualification."""

import argparse
import json
from pathlib import Path

from check_resident_add32_full import RESULTS, cases_by_name, load_report

LAYOUT = "qwengpu-resident-tilebanks-9b-v1"


def check_tile(report: dict) -> None:
    old = load_report("resident_add32_cached_tile_mps")
    assert report["input_sha256"] == old["input_sha256"]
    assert report["preloaded_operands"] is True
    for index, (observed, baseline) in enumerate(
        zip(report["passes"], old["passes"], strict=True)
    ):
        for key in (
            "k",
            "matrix",
            "instructions",
            "scheduler_decisions",
            "trace_sha256",
        ):
            assert observed[key] == baseline[key]
        first = index == 0
        initial = 256 if first else 0
        audit = observed["memory_tokens_by_call"]
        transport = observed["transport"]
        assert audit[0] == {
            "operands": initial,
            "alternate_operands": initial,
            "c": initial,
        }
        assert sum(call["operands"] for call in audit) == initial
        assert sum(call["alternate_operands"] for call in audit) == initial
        assert sum(call["c"] for call in audit) == initial + 256
        assert transport["matrix_initial_tokens"] == 3 * initial
        assert transport["matrix_patch_tokens"] == 256
        assert transport["fresh_caches"] == int(first)
        assert transport["resident_operand_banks_exact"]
        assert transport["input_tokens"] == (
            baseline["transport"]["input_tokens"] + (256 if first else -256)
        )
    assert report["cold_second_slice"] == old["cold_second_slice"]
    assert (
        report["cold_second_slice"]["transport"]["input_tokens"]
        - report["passes"][1]["transport"]["input_tokens"]
        == 512
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="*", type=Path)
    parser.add_argument(
        "--full", action="store_true", help="check the inherited suites"
    )
    args = parser.parse_args()
    paths = args.reports or [
        RESULTS / f"resident_tilebanks_cached_tile_{backend}.json"
        for backend in ("mps", "9b_cuda_fla")
    ]
    previous = None
    for path in paths:
        report = json.loads(path.read_text())
        check_tile(report)
        if report["merge"] is not None:
            assert report["merge"]["layout"] == LAYOUT
        if previous is not None:
            assert report["passes"] == previous["passes"]
        previous = report
        print(f"{path.name}: exact slices and zero second-pass memory uploads")
    if args.full:
        baseline = cases_by_name(load_report("resident_loop32_full_mps"))
        for backend in ("mps", "9b_cuda_fla"):
            report = load_report(f"resident_tilebanks_full_{backend}")
            assert cases_by_name(report) == baseline
            if report["merge"] is not None:
                assert report["merge"] == previous["merge"]
            print(f"{backend}: exact records for all 27 inherited cases")


if __name__ == "__main__":
    main()
