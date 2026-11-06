"""Check high-bit divergence against scalar sums and inherited traces."""

import argparse
import hashlib
import json
from pathlib import Path

from check_resident_add32_full import RESULTS, cases_by_name, load_report

MANIFEST = (
    RESULTS.parent / "eval/frozen/resident_branch32_gate_v1/cooperative_branch32.json"
)
LAYOUT = "qwengpu-resident-branch32-9b-v1"


def check_branch(case: dict) -> None:
    payload = MANIFEST.read_bytes()
    fixture = json.loads(payload)
    assert case["fixture_sha256"] == hashlib.sha256(payload).hexdigest()
    assert case["outcome"] == "halted"
    assert case["instructions"] == 21 and len(case["schedules"]) == 22
    operands = bytes(fixture["a"] + fixture["b"])
    words = [
        int.from_bytes(operands[i : i + 2], "little")
        for i in range(0, len(operands), 2)
    ] + fixture["c"]
    active = then_mask = 0
    producer = []
    consumer = []
    for lane, seed in enumerate(fixture["initial_registers"][0]):
        base, rhs = seed & 127, (seed & 127) ^ 126
        left = words[base] | (words[base ^ 1] << 16)
        right = words[rhs] | (words[rhs ^ 1] << 16)
        total = (left + right) & 0xFFFFFFFF
        assert total & 65535 == 1
        if seed & 128:
            active |= 1 << lane
            sign = total >> 31
            then_mask |= sign << lane
            value = total + 6 - sign
            words[base], words[base ^ 1] = value & 65535, value >> 16
            producer.append(value)
            consumer.append(value)
        else:
            producer.append(seed)
            consumer.append(left)
    assert 0 < then_mask < active < 255
    assert case["warp_registers"] == fixture["expected_warp_registers"]
    assert case["warp_registers"] == [producer, consumer]
    assert case["shared_words"] == words
    assert case["transport"]["matrix_patch_tokens"] == 56

    steps = [step for step in case["trace"] if step["warp"] == 0]
    masks = [(step["mask"][0] & 15) | ((step["mask"][1] & 15) << 4) for step in steps]
    assert masks[5:12] == [
        active,
        then_mask,
        then_mask,
        active ^ then_mask,
        active ^ then_mask,
        active,
        active,
    ]
    for lane in range(8):
        if active >> lane & 1:
            digits = steps[5]["registers"][4 * lane : 4 * lane + 4]
            assert [token & 15 for token in digits] == [1, 0, 0, 0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="*", type=Path)
    args = parser.parse_args()
    paths = args.reports or [
        RESULTS / f"resident_branch32_full_{backend}.json"
        for backend in ("mps", "9b_cuda_fla")
    ]
    baseline = cases_by_name(load_report("resident_addmem32_full_mps"))
    previous = None
    for path in paths:
        report = json.loads(path.read_text())
        cases = cases_by_name(report)
        check_branch(cases.pop(MANIFEST.name))
        assert cases == baseline
        if report["merge"] is not None:
            assert report["merge"]["layout"] == LAYOUT
        if previous is not None:
            assert report["cases"] == previous
        previous = report["cases"]
        print(f"{path.name}: exact high-bit branch and all 25 inherited cases")


if __name__ == "__main__":
    main()
