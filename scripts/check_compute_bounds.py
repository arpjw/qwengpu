"""Check raw matrix reads outside the supported eight-row operand extent."""

from __future__ import annotations

import argparse
import hashlib
import json

from check_resident_add32_full import RESULTS, load_report

FIXTURE = RESULTS.parent / "eval/frozen/compute_bounds_v1/matrix_extent.json"


def check(report: dict, fixture=FIXTURE) -> None:
    payload = fixture.read_bytes()
    inputs = json.loads(payload)
    assert report["kernel_sha256"] == hashlib.sha256(payload).hexdigest()
    assert report["global_words"] == inputs["global_words"]
    assert report["instructions"] == len(report["trace"]) == 24
    assert len(report["schedules"]) == 25 and len(report["workgroup_transitions"]) == 3
    assert report["outcome"] == "halted" and report["workgroup_cursor"] == 3
    assert all(report["reference"].values())
    assert report["transport"]["fresh_caches"] == 1
    assert report["transport"]["matrix_initial_tokens"] == 4 * len(
        inputs["global_words"]
    )
    for index, step in enumerate(report["trace"]):
        if index % 6 not in (2, 3):
            continue
        ids = [16 * step["workgroup"] + 8 * step["warp"] + lane for lane in range(8)]
        expected = [value + 765 if value < 8 else value for value in ids]
        words = [
            sum(
                (token & 15) << (4 * digit)
                for digit, token in enumerate(step["registers"][start : start + 4])
            )
            for start in range(0, 32, 4)
        ]
        assert words == expected
    assert report["warp_registers"] == [list(range(48, 56)), list(range(56, 64))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    mps = load_report("compute_bounds_mps")
    check(mps)
    assert mps["checkpoint"] == "standalone-resident-compute"
    print("MPS: raw matrix instruction preserves all out-of-extent lanes and memory")
    if args.cuda:
        cuda = load_report("compute_bounds_9b_cuda_fla")
        check(cuda)
        assert cuda["checkpoint"] == "qwengpu-resident-compute-9b-v1"
        for field in (
            "trace",
            "schedules",
            "global_words",
            "warp_registers",
            "transport",
            "workgroup_transitions",
        ):
            assert cuda[field] == mps[field], field
        print("Merged 9B: all four workgroups exactly match MPS, including raw traces")


if __name__ == "__main__":
    main()
