"""Verify shared32 handoff and replay every inherited ADD32 case."""

import argparse
import hashlib
import json
from pathlib import Path

from check_resident_add32_full import RESULTS, cases_by_name, load_report

MANIFEST = (
    RESULTS.parent / "eval/frozen/resident_shared32_gate_v1/cooperative_shared32.json"
)
LAYOUT = "qwengpu-resident-shared32-9b-v1"


def check_handoff(case: dict) -> None:
    payload = MANIFEST.read_bytes()
    manifest = json.loads(payload)
    assert case["fixture_sha256"] == hashlib.sha256(payload).hexdigest()
    expected = manifest["expected_warp_registers"]
    assert case["outcome"] == "halted"
    assert case["instructions"] == 20 and len(case["schedules"]) == 21
    assert case["warp_registers"] == expected

    operands = bytes(manifest["a"] + manifest["b"])
    words = [
        int.from_bytes(operands[i : i + 2], "little")
        for i in range(0, len(operands), 2)
    ] + manifest["c"]
    for lane in (0, 1, 3, 4, 5, 6):
        addr = manifest["initial_registers"][0][lane] & 127
        value = expected[0][lane]
        words[addr] = value & 65535
        words[addr ^ 1] = value >> 16
    assert case["shared_words"] == words

    stores = [step["patch"] for step in case["trace"] if any(step["patch"])]
    assert len(stores) == 2
    for patch in stores:
        assert [lane for lane in range(8) if all(patch[4 * lane : 4 * lane + 4])] == [
            0,
            1,
            3,
            4,
            5,
            6,
        ]
        assert sum(bool(token) for token in patch) == 24
    assert case["transport"]["matrix_patch_tokens"] == 48


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="*", type=Path)
    args = parser.parse_args()
    reports = args.reports or [
        RESULTS / f"resident_shared32_full_{backend}.json"
        for backend in ("mps", "9b_cuda_fla")
    ]
    inherited = cases_by_name(load_report("resident_add32_full_mps"))
    previous = None
    for path in reports:
        report = json.loads(path.read_text())
        cases = cases_by_name(report)
        check_handoff(cases.pop(MANIFEST.name))
        assert cases == inherited
        if report["merge"] is not None:
            assert report["merge"]["layout"] == LAYOUT
        if previous is not None:
            assert report["cases"] == previous["cases"]
        previous = report
        print(f"{path.name}: exact 32-bit handoff and all 23 inherited cases")


if __name__ == "__main__":
    main()
