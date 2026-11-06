"""Compare the frozen 16×16 launch with independent scalar products."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "eval/frozen/matmul16_gate_v1/dense_seeded.json"
RESULTS = ROOT / "results"


def check_report(report: dict, inputs: dict, digest: str) -> None:
    assert report["format"] == "qwengpu-matmul16-v1"
    assert report["input_sha256"] == digest
    assert report["transport_mode"] == "fixed"
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
            assert transport["fresh_caches"] == 1
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
        add32 = "checkpoint" in report
        if report["merge"]:
            baseline = "resident_add32" if add32 else "resident_grid"
            prior = json.loads((RESULTS / f"{baseline}_9b_cuda_fla.json").read_text())
            assert report["merge"] == prior["merge"]
        if add32:
            expected_layout = (
                report["merge"]["layout"]
                if report["merge"]
                else "standalone-resident-add32"
            )
            assert report["checkpoint"] == expected_layout
            check_grid_parity(report)
        if add32 in previous:
            assert report["passes"] == previous[add32]["passes"]
        previous[add32] = report
        message = (
            f"{path.name}: both passes match the scalar reference, 928 instructions"
        )
        if add32:
            message += ", all eight projected traces match the grid"
        print(message)


if __name__ == "__main__":
    main()
