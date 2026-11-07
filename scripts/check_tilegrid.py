"""Verify the saved one-cache grid against the qualified four-cache baseline."""

from __future__ import annotations

import argparse
import hashlib
import json

from check_matmul16 import FIXTURE
from check_resident_add32_full import cases_by_name, load_report
from check_tilepipe_matmul16 import check_text

LAYOUT = "qwengpu-resident-tilegrid-9b-v1"
ROOT = FIXTURE.parents[3]


def check(report: dict, baseline: dict) -> None:
    assert report["fixture_sha256"] == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert report["grid_2x2"] and report["row_pairs"] == 4 and report["exact"]
    if report["merge"]:
        assert report["merge"]["layout"] == LAYOUT
    assert report["instructions"] == 928
    assert report["scheduler_decisions"] == 929
    assert [(item["workgroup"], item["bank"]) for item in report["passes"]] == [
        (group, bank) for group in range(4) for bank in range(2)
    ]
    for bank in range(2):
        tiles = [item["matrix"] for item in report["passes"] if item["bank"] == bank]
        matrix = [
            left + right
            for row in (0, 2)
            for left, right in zip(tiles[row], tiles[row + 1], strict=True)
        ]
        assert matrix == baseline["passes"][bank]["matrix"]
    assert report["matrix"] == matrix == baseline["matrix"]
    assert [item["ticket"] for item in report["transitions"]] == [
        1632,
        1696,
        1632,
        1760,
        1632,
        1952,
        1632,
    ]
    assert all(item["resident_memory_exact"] for item in report["transitions"])
    transport = report["transport"]
    assert transport["fresh_caches"] == transport["grid_initial_tokens"] == 1
    assert (
        transport["matrix_initial_tokens"] == transport["matrix_patch_tokens"] == 2048
    )
    assert (
        transport["rom_program_tokens"] == 32
        and transport["warp_initial_tokens"] == 192
    )
    assert all(
        transport[f"resident_{name}_exact"]
        for name in ("grid", "memory", "operand_banks", "rom", "state")
    )
    calls = report["memory_tokens_by_call"]
    assert len(calls) == transport["forward_calls"]
    assert calls[0] == {
        "operands": 256,
        "alternate_operands": 256,
        "right_operands": 256,
        "bottom_operands": 256,
        "c": 256,
        "second_output": 256,
        "bottom_outputs": 512,
        "grid": 1,
        "transition": 0,
        "workgroup_transition": 0,
    }
    upload_only = (
        "operands",
        "alternate_operands",
        "right_operands",
        "bottom_operands",
        "grid",
    )
    assert all(not any(call[name] for name in upload_only) for call in calls[1:])
    assert sum(call["transition"] for call in calls) == 4
    assert sum(call["workgroup_transition"] for call in calls) == 3


def check_programs(layout: dict | None) -> None:
    def words32(words: list[int]) -> list[int]:
        return [lo | hi << 16 for lo, hi in zip(words[::2], words[1::2], strict=True)]

    assembled = load_report("assembled_cooperative_pair_mps")
    vector = load_report("tilegrid_vector_mps")
    reduction = load_report("tilegrid_reduction_mps")
    generated = load_report("tilegrid_generated_mps")
    initial = json.loads((ROOT / "examples/guarded_vector_add_data.json").read_text())
    left, right = words32(initial["c"][:32]), words32(initial["c"][32:])
    expected = [
        (a + b) & 0xFFFFFFFF for a, b in zip(left[:13], right[:13], strict=True)
    ]
    assert words32(vector["shared_words"][64:]) == expected + left[13:] + right
    assert all(vector["reference"].values())
    assert (
        reduction["exact"] and reduction["sum"] == sum(reduction["input"]) & 0xFFFFFFFF
    )
    assert all(
        case["decisions"] == 33 and case["instructions"] == 32
        for case in reduction["passes"]
    )
    first, second = reduction["passes"]
    assert first["transport"]["fresh_caches"] == 1
    assert (
        second["transport"]["fresh_caches"]
        == second["transport"]["matrix_initial_tokens"]
        == 0
    )
    assert len(cases_by_name(generated)) == 6
    print("MPS: exact vector add, retained-cache reduction, and six seeded cases")
    if layout is None:
        return
    for name, report in (("assembled", assembled), ("vector", vector)):
        cuda = load_report(f"tilegrid_{name}_9b_cuda_fla")
        assert cuda["checkpoint"] == LAYOUT
        for field in (
            "trace",
            "schedules",
            "warp_registers",
            "shared_words",
            "reference",
        ):
            assert cuda[field] == report[field], (name, field)
    cuda_reduction = load_report("tilegrid_reduction_9b_cuda_fla")
    assert cuda_reduction["merge"] == layout
    for field in ("input_sha256", "sum", "passes", "guards"):
        assert cuda_reduction[field] == reduction[field], field
    cuda_generated = load_report("tilegrid_generated_9b_cuda_fla")
    assert cuda_generated["merge"] == layout
    assert cases_by_name(cuda_generated) == cases_by_name(generated)
    print("Merged 9B CUDA: exact assembly, vector, reduction, and seeded traces")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--inherited", action="store_true")
    parser.add_argument("--text", action="store_true")
    parser.add_argument("--kernels", action="store_true")
    args = parser.parse_args()
    if args.text and not args.cuda:
        parser.error("--text requires --cuda")
    baseline = load_report("resident_tilepipe_matmul16_mps")
    mps = load_report("tilegrid_2x2_full_mps")
    check(mps, baseline)
    previous_tokens = sum(
        tile["transport"]["input_tokens"] for tile in baseline["tiles"]
    )
    tokens = mps["transport"]["input_tokens"]
    print(
        f"MPS: exact, one cache, {tokens:,} input tokens (baseline {previous_tokens:,})"
    )
    if args.cuda:
        cuda = load_report("tilegrid_2x2_full_9b_cuda_fla")
        check(cuda, baseline)
        for field in (
            "matrix",
            "passes",
            "transitions",
            "transport",
            "memory_tokens_by_call",
        ):
            assert mps[field] == cuda[field], field
        print("Merged 9B CUDA: exact traces, matrices, and transport records")
    if args.inherited:
        previous = cases_by_name(load_report("resident_tilepipe_full_mps"))
        current = cases_by_name(load_report("tilegrid_inherited_mps"))
        assert len(current) == 27 and current == previous
        print("MPS: all 27 inherited records exact")
        if args.cuda:
            inherited = load_report("tilegrid_inherited_9b_cuda_fla")
            assert inherited["merge"] == cuda["merge"]
            assert cases_by_name(inherited) == current
            print("Merged 9B CUDA: all 27 inherited records exact")
    if args.text:
        check_text(cuda["merge"], "tilegrid")
    if args.kernels:
        check_programs(cuda["merge"] if args.cuda else None)


if __name__ == "__main__":
    main()
