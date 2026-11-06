"""Check the retained tile reports against the earlier composed multiply."""

import argparse
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="*", type=Path)
    args = parser.parse_args()
    paths = args.reports or [
        RESULTS / f"resident_add32_cached_tile_{backend}.json"
        for backend in ("mps", "9b_cuda_fla")
    ]
    baseline = json.loads((RESULTS / "resident_add32_matmul16_mps.json").read_text())
    previous = None
    for path in paths:
        report = json.loads(path.read_text())
        assert report["input_sha256"] == baseline["input_sha256"]
        if report["merge"]:
            merged = json.loads(
                (RESULTS / "resident_add32_matmul16_9b_cuda_fla.json").read_text()
            )
            assert report["merge"] == merged["merge"]
        for step, old in zip(report["passes"], baseline["passes"], strict=True):
            assert step["k"] == old["k"]
            assert step["matrix"] == [row[:8] for row in old["matrix"][:8]]
            assert step["trace_sha256"] == old["launches"][0]["trace_sha256"]
            assert step["instructions"] == 116 and step["scheduler_decisions"] == 117
        first, second = report["passes"]
        assert first["memory_tokens_by_call"][0] == {"operands": 256, "c": 256}
        assert second["memory_tokens_by_call"][0] == {"operands": 256, "c": 0}
        assert sum(call["operands"] for call in second["memory_tokens_by_call"]) == 256
        assert (
            sum(call["c"] for call in second["memory_tokens_by_call"])
            == second["transport"]["matrix_patch_tokens"]
        )
        assert second["transport"]["fresh_caches"] == 0
        cold = report["cold_second_slice"]
        assert (
            cold["trace_sha256"] == second["trace_sha256"]
            and cold["matrix"] == second["matrix"]
        )
        assert (
            cold["transport"]["input_tokens"] - second["transport"]["input_tokens"]
            == 256
        )
        if previous:
            assert report["passes"] == previous["passes"]
            assert cold == previous["cold_second_slice"]
        previous = report
        print(f"{path.name}: two exact slices and 256 C upload tokens saved")


if __name__ == "__main__":
    main()
