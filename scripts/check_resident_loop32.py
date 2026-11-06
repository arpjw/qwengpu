"""Verify upper-half loop counts and inherited checkpoint traces."""

import argparse
import hashlib
import json
from pathlib import Path

from check_resident_add32_full import RESULTS, cases_by_name, load_report

MANIFEST = (
    RESULTS.parent / "eval/frozen/resident_loop32_gate_v1/cooperative_loop32.json"
)
LAYOUT = "qwengpu-resident-loop32-9b-v1"


def check_loop(case: dict) -> None:
    payload = MANIFEST.read_bytes()
    fixture = json.loads(payload)
    assert case["fixture_sha256"] == hashlib.sha256(payload).hexdigest()
    assert case["outcome"] == "halted"
    assert case["instructions"] == 24 and len(case["schedules"]) == 25
    operands = bytes(fixture["a"] + fixture["b"])
    words = [
        int.from_bytes(operands[i : i + 2], "little")
        for i in range(0, len(operands), 2)
    ] + fixture["c"]
    parent = 0
    trips, producer, consumer = [], [], []
    for lane, seed in enumerate(fixture["initial_registers"][0]):
        base = seed & 127
        original = words[base] | (words[base ^ 1] << 16)
        value = original
        steps = 0
        if seed & 128:
            parent |= 1 << lane
            while value & 65536:
                value = (value + 1) & 0xFFFFFFFF
                steps += 1
            value = (value & 0xFFFF0000) | ((value + 7) & 65535)
            words[base], words[base ^ 1] = value & 65535, value >> 16
            producer.append(value)
            consumer.append(value)
        else:
            producer.append(seed)
            consumer.append(original)
        trips.append(steps)
    assert trips == fixture["expected_trip_counts"] == [3, 2, 1, 0, 1, 0, 0, 2]
    assert case["warp_registers"] == [producer, consumer]
    assert case["warp_registers"] == fixture["expected_warp_registers"]
    assert case["shared_words"] == words
    assert case["transport"]["matrix_patch_tokens"] == 56

    steps = [step for step in case["trace"] if step["warp"] == 0]
    observed = [
        (step["mask"][0] & 15) | ((step["mask"][1] & 15) << 4) for step in steps
    ]
    expected = [255, parent, parent, parent]
    for iteration in range(max(trips)):
        selected = sum(1 << lane for lane, trip in enumerate(trips) if trip > iteration)
        expected.extend((selected, selected, parent))
    expected.extend([parent] * 6)
    assert observed == expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="*", type=Path)
    args = parser.parse_args()
    paths = args.reports or [
        RESULTS / f"resident_loop32_full_{backend}.json"
        for backend in ("mps", "9b_cuda_fla")
    ]
    baseline = cases_by_name(load_report("resident_branch32_full_mps"))
    previous = None
    for path in paths:
        report = json.loads(path.read_text())
        cases = cases_by_name(report)
        check_loop(cases.pop(MANIFEST.name))
        assert cases == baseline
        if report["merge"] is not None:
            assert report["merge"]["layout"] == LAYOUT
        if previous is not None:
            assert report["cases"] == previous
        previous = report["cases"]
        print(f"{path.name}: exact upper-half loops and all 26 inherited cases")


if __name__ == "__main__":
    main()
