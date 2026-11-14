"""Verify resident generation reports against the saved cache-adapter baseline."""

import argparse
import json
from pathlib import Path


def check(directory, reports):
    for path in reports:
        report = json.loads(path.read_text())
        for case in report["cases"]:
            assert all(value for key, value in case.items() if key.endswith("_exact")), case
            assert case["generated_tokens"] == case["expected_tokens"]
            assert case["generation_calls"] == 1
            assert case.get("host_execution_tokens", 0) == 0
            fixture = Path(case["fixture"]).stem
            if fixture == "cross_group_high_limb":
                assert case["global_words"][:4] == [71, 1, 72, 1]
                assert case["steps"] == 48
            else:
                baseline_path = directory / f"merged_{fixture}.json"
                baseline = json.loads(baseline_path.read_text())
                assert case["sha256"] == baseline["kernel_sha256"]
                assert case["steps"] == baseline["instructions"]
                assert case["outcome"] == baseline["outcome"]
                assert case["global_words"] == baseline["global_words"]
                assert case["scoped_words"] == baseline.get("scoped_words")
            print(f"{path.name} {fixture}: {case['steps']} instructions, "
                  f"{len(case['generated_tokens'])} exact generated tokens")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("reports", type=Path, nargs="+")
    args = parser.parse_args()
    check(args.baseline, args.reports)
