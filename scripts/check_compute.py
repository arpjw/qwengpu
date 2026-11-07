"""Check saved common-kernel results against scalar outputs and memory guards."""

from __future__ import annotations

import argparse
import hashlib
import json

from check_resident_add32_full import RESULTS, load_report

FIXTURE = RESULTS.parent / "eval/frozen/compute_v1/cases.json"
LAYOUT = "qwengpu-resident-compute-9b-v1"
GUARD = 0xD00D0000


def limbs(values: list[int]) -> list[int]:
    return [part for value in values for part in (value & 65535, value >> 16)]


def check(report: dict, names: set[str], fixture=FIXTURE) -> None:
    payload = fixture.read_bytes()
    inputs = json.loads(payload)
    assert report["fixture_sha256"] == hashlib.sha256(payload).hexdigest()
    assert report["all_exact"]
    records = {record["name"]: record for record in report["cases"]}
    assert len(records) == len(report["cases"]) and records.keys() == names
    if report["merge"]:
        assert report["merge"]["layout"] == LAYOUT
    for name, record in records.items():
        request = inputs[name]
        assert record["request"] == request
        kind = request["kernel"]
        if kind == "vector_add":
            a, b = request["a"], request["b"]
            count = len(a)
            expected_output = [(x + y) & 0xFFFFFFFF for x, y in zip(a, b, strict=True)]
            expected_words = limbs(
                expected_output
                + [GUARD + i for i in range(32 - count)]
                + b
                + [GUARD + 32 + i for i in range(32 - count)]
            )
        elif kind == "reduce":
            values = request["values"]
            expected_output = sum(values) & 0xFFFFFFFF
            padded = max(32, len(values))
            expected_words = limbs(
                values
                + [0] * (padded - len(values))
                + [GUARD + i for i in range(64 - padded)]
            )
            for start in range(0, len(values), 8):
                expected_words[2 * start : 2 * start + 2] = limbs(
                    [sum(values[start : start + 8]) & 0xFFFFFFFF]
                )
            expected_words[:2] = limbs([expected_output])
        else:
            a, b, c = request["a"], request["b"], request["c"]
            m, k, n = len(a), len(b), len(b[0])
            expected_output = [
                [
                    (c[i][j] + sum(a[i][t] * b[t][j] for t in range(k))) & 0xFFFFFFFF
                    for j in range(n)
                ]
                for i in range(m)
            ]
            expected_words = [v for row in a for v in row] + [
                0xAB00 + i for i in range(32 - m * k)
            ]
            expected_words += [v for row in b for v in row] + [
                0xBC00 + i for i in range(16 - k * n)
            ]
            expected_words += [0xCD00 + i for i in range(16)]
            expected_words += limbs(
                [v for row in expected_output for v in row]
                + [GUARD + i for i in range(32 - m * n)]
            )
        assert record["output"] == expected_output, name
        assert record["global_words"] == expected_words, name
        assert record["exact"]
        runs = record["launches"]
        assert len(runs) == (2 if kind == "reduce" else 1)
        for index, run in enumerate(runs):
            transport = run["transport"]
            assert transport["fresh_caches"] == int(index == 0)
            assert transport["matrix_initial_tokens"] == (512 if index == 0 else 0)
            assert all(
                transport[f"resident_{field}_exact"]
                for field in ("grid", "memory", "rom", "state")
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--extent", action="store_true")
    args = parser.parse_args()
    if sum((args.smoke, args.decode, args.extent)) > 1:
        parser.error("choose smoke, decode, or extent")
    fixture = (
        FIXTURE.parent.parent / "compute_decode_v1/cases.json"
        if args.decode
        else FIXTURE.parent.parent / "compute_extent_v1/cases.json"
        if args.extent
        else FIXTURE
    )
    names = (
        {"vector_17", "reduce_13", "matmul_3x3x3"}
        if args.smoke
        else set(json.loads(fixture.read_bytes()))
    )
    stem = (
        "compute_decode"
        if args.decode
        else "compute_extent"
        if args.extent
        else "compute"
    )

    def selected(report: dict) -> dict:
        return (
            {
                **report,
                "cases": [case for case in report["cases"] if case["name"] in names],
            }
            if args.smoke
            else report
        )

    mps = selected(load_report(f"{stem}_mps"))
    check(mps, names, fixture)
    print(f"MPS: {len(names)} compute cases exact, including all untouched memory")
    if args.cuda:
        cuda = selected(load_report(f"{stem}_9b_cuda_fla"))
        check(cuda, names, fixture)
        assert cuda["cases"] == mps["cases"]
        print("Merged 9B CUDA: outputs, traces, cache state, and transport match MPS")


if __name__ == "__main__":
    main()
