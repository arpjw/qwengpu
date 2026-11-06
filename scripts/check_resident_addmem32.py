"""Verify full-width memory addition and replay the shared32 cases."""

import argparse
import hashlib
import json
from pathlib import Path

from check_resident_add32_full import RESULTS, cases_by_name, load_report

MANIFEST = (
    RESULTS.parent / "eval/frozen/resident_addmem32_gate_v1/cooperative_addmem32.json"
)
LAYOUT = "qwengpu-resident-addmem32-9b-v1"


def check_sum(case: dict) -> None:
    payload = MANIFEST.read_bytes()
    fixture = json.loads(payload)
    assert case["fixture_sha256"] == hashlib.sha256(payload).hexdigest()
    assert case["outcome"] == "halted"
    assert case["instructions"] == 15 and len(case["schedules"]) == 16
    operands = bytes(fixture["a"] + fixture["b"])
    words = [
        int.from_bytes(operands[i : i + 2], "little")
        for i in range(0, len(operands), 2)
    ] + fixture["c"]
    producer = []
    consumer = []
    for seed in fixture["initial_registers"][0]:
        base = seed & 127
        rhs = base ^ 126
        left = words[base] | (words[base ^ 1] << 16)
        right = words[rhs] | (words[rhs ^ 1] << 16)
        value = (left + right) & 0xFFFFFFFF
        if seed & 128:
            words[base], words[base ^ 1] = value & 65535, value >> 16
            producer.append(value)
            consumer.append(value)
        else:
            producer.append(seed)
            consumer.append(left)
    assert case["warp_registers"] == fixture["expected_warp_registers"]
    assert case["warp_registers"] == [producer, consumer]
    assert case["shared_words"] == words
    assert case["transport"]["matrix_patch_tokens"] == 56


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="*", type=Path)
    args = parser.parse_args()
    reports = args.reports or [
        RESULTS / f"resident_addmem32_full_{backend}.json"
        for backend in ("mps", "9b_cuda_fla")
    ]
    baseline = cases_by_name(load_report("resident_shared32_full_mps"))
    previous = None
    for path in reports:
        report = json.loads(path.read_text())
        cases = cases_by_name(report)
        check_sum(cases.pop(MANIFEST.name))
        assert cases == baseline
        if report["merge"] is not None:
            assert report["merge"]["layout"] == LAYOUT
        if previous is not None:
            assert report["cases"] == previous
        previous = report["cases"]
        print(f"{path.name}: exact shared-operand sums and all 24 inherited cases")


if __name__ == "__main__":
    main()
