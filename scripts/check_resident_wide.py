"""Compare resident high-register snapshots and launch transports."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIXTURES = ROOT / "eval/frozen"
LEGACY_SNAPSHOT_DIGITS = 65
CANONICAL = ("outcome", "instructions", "matrix", "shared_words", "trace", "schedules")


def read_result(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def check_suite(report: dict) -> dict[str, dict]:
    assert report["all_exact"]
    cases = {case["fixture"]: case for case in report["cases"]}
    old = {
        case["fixture"]: case for case in read_result("resident_grid_mps.json")["cases"]
    }
    assert cases.keys() == {
        "seeded_warps.json",
        "seeded_cooperative.json",
        "dense_seeded.json",
    }
    for name, case in cases.items():
        folder = "tile8_gate_v1" if name == "dense_seeded.json" else "wide_gate_v1"
        payload = (FIXTURES / folder / name).read_bytes()
        assert case["fixture_sha256"] == hashlib.sha256(payload).hexdigest()
        seed = json.loads(payload).get("initial_registers", [[0] * 8, [0] * 8])
        high = [[word >> 16 for word in warp] for warp in seed]
        transport = case["transport"]
        assert (
            transport["fresh_caches"] == 1 and transport["warp_initial_tokens"] == 192
        )
        assert transport["state_plan_tokens"] == 12 * case["instructions"]
        assert transport["store_plan_tokens"] == case["instructions"]
        assert all(
            transport[f"resident_{bank}_exact"]
            for bank in ("grid", "memory", "rom", "state")
        )
        assert [
            [word >> 16 for word in warp] for warp in case["warp_registers"]
        ] == high
        for tick in case["trace"]:
            packed = int(tick["snapshot"][:-LEGACY_SNAPSHOT_DIGITS], 16)
            assert [(packed >> (16 * lane)) & 65535 for lane in range(8)] == high[
                tick["warp"]
            ]

    for name, before in (
        ("dense_seeded.json", old["dense_seeded.json"]),
        ("seeded_cooperative.json", old["cooperative_matmul.json"]),
    ):
        case = cases[name]
        assert all(case[key] == before[key] for key in CANONICAL if key != "trace")
        assert [
            {**tick, "snapshot": tick["snapshot"][-LEGACY_SNAPSHOT_DIGITS:]}
            for tick in case["trace"]
        ] == before["trace"]
    return cases


def main() -> None:
    mps = read_result("resident_wide_mps.json")
    cuda = read_result("resident_wide_9b_cuda_fla.json")
    cases = check_suite(mps)
    assert cuda["all_exact"] and cuda["cases"] == mps["cases"]
    assert cuda["merge"]["layout"] == "qwengpu-resident-wide9b-v1"

    case = cases["seeded_warps.json"]
    for mode in ("fixed", "state-plan"):
        for backend, layout in (
            ("mps", "standalone-resident-wide"),
            ("9b_cuda_fla", cuda["merge"]["layout"]),
        ):
            launch = read_result(f"resident_wide_launch_{mode}_{backend}.json")
            assert launch["checkpoint"] == layout
            assert launch["transport_mode"] == mode
            assert launch["kernel_sha256"] == case["fixture_sha256"]
            assert launch["transport"]["resident_state_exact"]
            assert all(launch[key] == case[key] for key in (*CANONICAL, "warp_registers"))
    print("Three exact cases across MPS and merged CUDA, with 274 high-limb snapshots")
    print(
        "Dense and cooperative traces match the prior grid; all three transports agree"
    )


if __name__ == "__main__":
    main()
