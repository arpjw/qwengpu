"""Check four model-scheduled tiles and the merged text boundary."""

import argparse
import hashlib
import json
from pathlib import Path

from check_addmem32_text import check_reset
from check_matmul16 import FIXTURE
from check_resident_add32_full import RESULTS, load_report
from check_tilepipe import LAYOUT, NEXT_BANK


def check_matmul(report: dict) -> None:
    fixture = FIXTURE.read_bytes()
    assert report["input_sha256"] == hashlib.sha256(fixture).hexdigest()
    assert report["schedule_k"] is True
    assert report["transport_mode"] == "state_store_plan"
    assert report["checkpoint"] == (
        LAYOUT if report["merge"] else "standalone-resident-tilepipe"
    )
    if report["merge"] is not None:
        assert (
            report["merge"]
            == load_report("resident_tilepipe_full_9b_cuda_fla")["merge"]
        )
    inputs = json.loads(fixture)
    a, b, c = (inputs[key] for key in ("a", "b", "c"))
    baseline = load_report("resident_add32_matmul16_retained_mps")
    for k, actual, old in zip(
        (0, 8), report["passes"], baseline["passes"], strict=True
    ):
        assert actual["k"] == k
        scalar = [
            [
                (c[row][col] + sum(a[row][i] * b[i][col] for i in range(k + 8))) % 65536
                for col in range(16)
            ]
            for row in range(16)
        ]
        assert actual["matrix"] == scalar == old["matrix"]
    assert report["matrix"] == report["passes"][-1]["matrix"]
    assert [(t["row"], t["col"]) for t in report["tiles"]] == [
        (0, 0),
        (0, 8),
        (8, 0),
        (8, 8),
    ]

    sample = load_report("resident_tilepipe_mps")
    for index, tile in enumerate(report["tiles"]):
        assert tile["instructions"] == 232 and tile["scheduler_decisions"] == 233
        row, col = tile["row"], tile["col"]
        for i, (step, old) in enumerate(
            zip(tile["passes"], baseline["passes"], strict=True)
        ):
            assert step["k"] == old["k"]
            assert step["instructions"] == 116 and step["scheduler_decisions"] == 117
            assert (
                step["separate_trace_sha256"] == old["launches"][index]["trace_sha256"]
            )
            assert step["matrix"] == [
                line[col : col + 8]
                for line in report["passes"][i]["matrix"][row : row + 8]
            ]
        first, second = tile["passes"]
        assert first["trace_sha256"] == first["separate_trace_sha256"]
        assert first["register_carry_differences"] == []
        assert second["register_carry_differences"] == [
            {"instruction": 1, "fields": ["registers", "snapshot"]}
        ]
        assert tile["bank_transition"] == {
            "ticket": NEXT_BANK,
            "instructions": 116,
            "schedule_index": 116,
            "matrix": first["matrix"],
            "resident_memory_exact": True,
            "bank_before": 0,
        }
        transport = tile["transport"]
        assert transport["fresh_caches"] == transport["grid_initial_tokens"] == 1
        assert (
            transport["rom_program_tokens"] == 32
            and transport["rom_select_tokens"] == 2
        )
        assert transport["warp_initial_tokens"] == 192
        assert transport["matrix_initial_tokens"] == 768
        assert transport["matrix_patch_tokens"] == 512
        assert all(
            transport[f"resident_{key}_exact"]
            for key in ("grid", "memory", "rom", "state", "operand_banks")
        )
        calls = tile["memory_tokens_by_call"]
        assert calls[0] == {
            "operands": 256,
            "alternate_operands": 256,
            "c": 256,
            "grid": 1,
            "transition": 0,
        }
        assert all(
            not any(call[key] for key in ("operands", "alternate_operands", "grid"))
            for call in calls[1:]
        )
        assert sum(call["transition"] for call in calls) == 1
        assert sum(call["c"] for call in calls[1:]) == 512
    for key in ("bank_transition", "passes", "transport", "memory_tokens_by_call"):
        assert report["tiles"][0][key] == sample[key]


def check_text(layout: dict) -> None:
    fresh, reset, roundtrip = (
        load_report(f"resident_tilepipe_{name}_9b_cuda_fla")
        for name in ("text", "handoff_reset", "arithmetic_relaunch")
    )
    assert all(report["merge"] == layout for report in (fresh, reset, roundtrip))
    old_fresh = load_report("resident_add32_text_9b_cuda_fla")
    for key in (
        "prompts_sha256",
        "tokenized_prompts_sha256",
        "max_new_tokens",
        "generation_config",
        "prompt_count",
    ):
        assert fresh[key] == old_fresh[key]
    assert len(fresh["records"]) == fresh["prompt_count"] == 24
    for key in ("next_top1_same", "greedy_identical", "merged_chip_tokens"):
        assert fresh[key] == sum(record[key] for record in fresh["records"])
    assert fresh["merged_chip_tokens"] == 0

    old_reset = load_report("resident_add32_handoff_reset_9b_cuda_fla")
    assert len(reset["records"]) == 4
    for record, baseline in zip(reset["records"], old_reset["records"], strict=True):
        for key in ("prompt_index", "prompt_ids", "gpu", "gpu_rows"):
            assert record[key] == baseline[key]
        check_reset(record["reset_vs_fresh"])
    old_relaunch = load_report("resident_addmem32_arithmetic_relaunch_9b_cuda_fla")
    assert roundtrip == {**old_relaunch, "merge": layout, "text": roundtrip["text"]}
    check_reset(roundtrip["text"]["comparison"])
    print(
        f"Text: {fresh['next_top1_same']}/24 first tokens, "
        f"{fresh['greedy_identical']}/24 greedy continuations; "
        "four exact resets and an exact full-width GPU relaunch"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="*", type=Path)
    parser.add_argument("--text", action="store_true")
    args = parser.parse_args()
    paths = args.reports or [
        RESULTS / f"resident_tilepipe_matmul16_{backend}.json"
        for backend in ("mps", "9b_cuda_fla")
    ]
    previous = None
    for path in paths:
        report = json.loads(path.read_text())
        check_matmul(report)
        if previous is not None:
            for key in ("matrix", "passes", "tiles"):
                assert report[key] == previous[key]
        previous = report
        print(f"{path.name}: four exact transitions and both scalar matrices")
    if args.text:
        check_text(previous["merge"])


if __name__ == "__main__":
    main()
