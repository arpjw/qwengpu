"""Compare the frozen 16×16 launch with independent scalar products."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "eval/frozen/matmul16_gate_v1/dense_seeded.json"
RESULTS = ROOT / "results"
SHARED32_LAYOUT = "qwengpu-resident-shared32-9b-v1"


def check_report(report: dict, inputs: dict, digest: str) -> None:
    assert report["format"] == "qwengpu-matmul16-v1"
    assert report["input_sha256"] == digest
    retain_c = report.get("retain_c", False)
    assert report["transport_mode"] == ("state_store_plan" if retain_c else "fixed")
    a, b, c = (inputs[key] for key in ("a", "b", "c"))
    assert [entry["k"] for entry in report["passes"]] == [0, 8]
    for step in report["passes"]:
        k_stop = step["k"] + 8
        scalar = [
            [
                (c[row][col] + sum(a[row][k] * b[k][col] for k in range(k_stop)))
                % 65536
                for col in range(16)
            ]
            for row in range(16)
        ]
        assert step["matrix"] == scalar, f"K={k_stop}"
        launches = step["launches"]
        assert [(tile["row"], tile["col"]) for tile in launches] == [
            (0, 0),
            (0, 8),
            (8, 0),
            (8, 8),
        ]
        for tile in launches:
            assert tile["instructions"] == 116 and tile["scheduler_decisions"] == 117
            transport = tile["transport"]
            new_cache = not retain_c or step["k"] == 0
            assert transport["fresh_caches"] == int(new_cache)
            if retain_c:
                initial_c = 256 * int(new_cache)
                audit = tile["memory_tokens_by_call"]
                assert audit[0] == {"operands": 256, "c": initial_c}
                assert sum(call["operands"] for call in audit) == 256
                assert sum(call["c"] for call in audit) == (
                    initial_c + transport["matrix_patch_tokens"]
                )
                assert transport["matrix_initial_tokens"] == 256 + initial_c
            assert all(
                transport[f"resident_{bank}_exact"]
                for bank in ("grid", "memory", "rom", "state")
            )
    assert report["matrix"] == report["passes"][-1]["matrix"]


def check_grid_parity(report: dict) -> None:
    grid = json.loads((RESULTS / "resident_matmul16_mps.json").read_text())
    assert report["input_sha256"] == grid["input_sha256"]
    for wide_pass, grid_pass in zip(report["passes"], grid["passes"], strict=True):
        assert wide_pass["matrix"] == grid_pass["matrix"]
        for wide, old in zip(wide_pass["launches"], grid_pass["launches"], strict=True):
            assert wide["grid_trace_sha256"] == old["trace_sha256"]
            assert all(
                wide[key] == old[key]
                for key in ("row", "col", "instructions", "scheduler_decisions")
            )
            transport = old["transport"]
            assert wide["transport"] == {
                **transport,
                "warp_initial_tokens": transport["warp_initial_tokens"] + 64,
                "input_tokens": transport["input_tokens"]
                + 64
                + 32 * wide["instructions"],
                "high_register_updates": 0,
            }


def check_retained_parity(report: dict, *, shared32: bool) -> None:
    backend = "9b_cuda_fla" if report["merge"] else "mps"
    baseline = json.loads(
        (RESULTS / f"resident_add32_matmul16_{backend}.json").read_text()
    )
    assert report["input_sha256"] == baseline["input_sha256"]
    if shared32:
        old_retained = json.loads(
            (RESULTS / f"resident_add32_matmul16_retained_{backend}.json").read_text()
        )
        assert report["passes"] == old_retained["passes"]
    else:
        for key in ("merge", "checkpoint"):
            assert report[key] == baseline[key]
    for step, old in zip(report["passes"], baseline["passes"], strict=True):
        assert step["matrix"] == old["matrix"]
        for tile, old_tile in zip(step["launches"], old["launches"], strict=True):
            for key in ("trace_sha256", "grid_trace_sha256"):
                assert tile[key] == old_tile[key]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="*")
    args = parser.parse_args()
    paths = args.reports or [
        RESULTS / "resident_matmul16_mps.json",
        RESULTS / "resident_matmul16_9b_cuda_fla.json",
    ]
    payload = FIXTURE.read_bytes()
    inputs = json.loads(payload)
    digest = hashlib.sha256(payload).hexdigest()
    previous = {}
    for path in paths:
        report = json.loads(path.read_text())
        check_report(report, inputs, digest)
        retain_c = report.get("retain_c", False)
        add32 = "checkpoint" in report
        shared32 = report.get("checkpoint") in (
            SHARED32_LAYOUT,
            "standalone-resident-shared32",
        )
        if report["merge"] and not shared32:
            baseline = "resident_add32" if add32 else "resident_grid"
            prior = json.loads((RESULTS / f"{baseline}_9b_cuda_fla.json").read_text())
            assert report["merge"] == prior["merge"]
        if add32:
            expected_layout = (
                report["merge"]["layout"]
                if report["merge"]
                else "standalone-resident-shared32"
                if shared32
                else "standalone-resident-add32"
            )
            assert report["checkpoint"] == expected_layout
            if retain_c:
                check_retained_parity(report, shared32=shared32)
            else:
                check_grid_parity(report)
        else:
            assert not retain_c
        group = (add32, retain_c, shared32)
        if group in previous:
            assert report["passes"] == previous[group]["passes"]
        previous[group] = report
        message = (
            f"{path.name}: both passes match the scalar reference, 928 instructions"
        )
        if retain_c:
            message += ", four fresh caches, 1,024 C upload tokens omitted"
        elif add32:
            message += ", all eight projected traces match the grid"
        print(message)


if __name__ == "__main__":
    main()
