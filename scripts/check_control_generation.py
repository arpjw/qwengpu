"""Compare stock-generation traces with the qualified host-clocked control trace."""

import argparse
import json
from pathlib import Path


def check(baseline: Path, reports: list[Path]) -> None:
    previous = json.loads(baseline.read_text())
    expected = {case["fixture"]: case for case in previous["cases"]}
    assert previous["all_exact"] and previous["all_expected_outcomes"]
    assert len(expected) == 14
    keys = ("stage", "executed_pc", "generated", "observed_snapshot_hex")

    for path in reports:
        report = json.loads(path.read_text())
        actual = {case["fixture"]: case for case in report["cases"]}
        assert report["all_exact"] and report["all_expected_outcomes"], path
        assert actual.keys() == expected.keys()
        assert len(report["cases"]) == len(expected)
        count = 0
        for name, case in actual.items():
            baseline_case = expected[name]
            assert case["fixture_sha256"] == baseline_case["fixture_sha256"], name
            assert case["outcome"] == baseline_case["outcome"], name
            assert case["generation_calls"] == 1, name
            assert case["host_tick_tokens"] == 0, name
            assert case["host_control_feedback_tokens"] == 0, name
            assert case["lane_feedback_tokens"] == 0, name
            for current, old in zip(case["trace"], baseline_case["trace"], strict=True):
                assert current["exact"] and all(current[key] == old[key] for key in keys), name
                count += 1
        print(f"{path.name}: {len(expected)} programs and {count} instruction records match")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("reports", type=Path, nargs="+")
    args = parser.parse_args()
    check(args.baseline, args.reports)
