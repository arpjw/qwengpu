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
    previous = None
    for path in paths:
        report = json.loads(path.read_text())
        check_report(report, inputs, digest)
        if report["merge"]:
            prior = json.loads((RESULTS / "resident_grid_9b_cuda_fla.json").read_text())
            assert report["merge"] == prior["merge"]
        if previous is not None:
            assert report["passes"] == previous["passes"]
        previous = report
        print(f"{path.name}: both passes match the scalar reference, 928 instructions")


if __name__ == "__main__":
    main()
