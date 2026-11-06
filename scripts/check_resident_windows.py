"""Check that resident windows compose into the same full tile."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIXTURES = ROOT / "eval" / "frozen"
CANONICAL = ("outcome", "instructions", "matrix", "shared_words", "trace", "schedules")


def read_result(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def check_windows() -> None:
    mps = read_result("resident_windows_mps.json")
    cuda = read_result("resident_windows_9b_cuda_fla.json")
    previous = read_result("resident_grid_9b_cuda_fla.json")
    assert mps["all_exact"] and cuda["all_exact"]
    assert mps["cases"] == cuda["cases"]
    assert cuda["merge"] == previous["merge"]

    cases = {case["fixture"]: case for case in mps["cases"]}
    full = cases.pop("dense_seeded.json")
    assert full == next(
        case for case in previous["cases"] if case["fixture"] == "dense_seeded.json"
    )
    inputs = json.loads((FIXTURES / "tile8_gate_v1" / "dense_seeded.json").read_text())
    initial = [inputs["c"][i : i + 8] for i in range(0, 64, 8)]

    windows = {
        "front_two.json": (0, 4),
        "back_two.json": (4, 8),
        "middle_one.json": (2, 4),
        "last_one.json": (6, 8),
    }
    assert cases.keys() == windows.keys()
    for name, (start, stop) in windows.items():
        payload = (FIXTURES / "window_gate_v1" / name).read_bytes()
        manifest = json.loads(payload)
        case = cases[name]
        assert all(manifest[key] == inputs[key] for key in ("a", "b", "c"))
        assert case["fixture_sha256"] == hashlib.sha256(payload).hexdigest()
        assert case["outcome"] == "halted"
        assert case["instructions"] == 29 * (stop - start) // 2
        assert case["matrix"][:start] == initial[:start]
        assert case["matrix"][start:stop] == full["matrix"][start:stop]
        assert case["matrix"][stop:] == initial[stop:]
        assert case["transport"]["fresh_caches"] == 1
        assert all(
            case["transport"][f"resident_{bank}_exact"]
            for bank in ("grid", "memory", "rom", "state")
        )

    front = cases["front_two.json"]["matrix"][:4]
    back = cases["back_two.json"]["matrix"][4:]
    assert front + back == full["matrix"]

    launch = read_result("resident_windows_launch_9b_cuda_fla.json")
    last = cases["last_one.json"]
    assert launch["checkpoint"] == cuda["merge"]["layout"]
    assert launch["kernel_sha256"] == last["fixture_sha256"]
    assert launch["first_row_pair"] == 3 and launch["row_pairs"] == 1
    assert launch["transport_mode"] == "fixed"
    assert all(launch[key] == last[key] for key in CANONICAL)
    print("Five exact cases across MPS and CUDA, with four row windows")
    print("Complementary windows reproduce all 64 full-tile outputs")


if __name__ == "__main__":
    check_windows()
